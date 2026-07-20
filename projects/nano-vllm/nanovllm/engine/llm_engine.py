import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.models.registry import load_tokenizer


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 问题（已回答）：max_model_len 指上下文长度还是一次输入 prompt 的长度？
        # 回答：它限制一条序列的总 token 数，即 prompt 加已经生成的 completion；并不只限制 prompt。
        # add_request 会据此预先检查“prompt 长度 + 最大生成长度”，避免位置编码和 KV cache 超出模型上限。
        self.max_model_len = config.max_model_len
        Sequence.block_size = config.kvcache_block_size
        # 问题（已回答）：这两个数组、spawn context 和 Event 分别做什么？
        # 回答：ps 保存 tensor-parallel worker 进程，events 保存主进程通知各 worker 的同步事件。
        # spawn 是 multiprocessing 的启动方式：每个子进程启动全新 Python 解释器，CUDA 场景比 fork 更安全。
        # Event 是跨进程信号；rank 0 写好共享内存命令后 set() 唤醒其他 rank。
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # 问题（已回答）：循环里只是定义进程，还没有启动吗？
        # 回答：ctx.Process(...) 只构造进程对象；紧接着 process.start() 已真正启动子进程并执行 ModelRunner。
        for i in range(1, config.tensor_parallel_size):
            # 问题（已回答）：这里的一个 Event 表示什么？
            # 回答：每个非零 rank 独占一个跨进程通知信号。rank 0 把命令写入共享内存后 set() 对应 Event，
            # 该 worker 从 wait() 醒来读取并执行；独立 Event 也让每个 worker 在读完后自行 clear()。
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 问题（已回答）：rank 0 ModelRunner 的作用是什么，为什么要区别不同 rank？
        # 回答：rank 0 位于主进程，既负责把同一 run/exit 命令广播给 worker、汇总输出并采样，也持有 GPU 0
        # 上的模型分片并参与每一次前向和集合通信；其余 rank 同样计算各自分片。rank 决定 GPU、参数 shard
        # 和 collective 中的职责，并不表示只有 rank 0 真正执行模型。
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = load_tokenizer(config.model, config.hf_config)
        # 问题（已回答）：分词器需要 eos_token_id 吗？
        # 回答：engine 需要统一读取它来判断生成何时结束。没有 EOS 的教学 tokenizer 可返回 None/-1，
        # 调用方再用 ignore_eos=True 或 max_tokens 作为停止条件。
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._exited = False
        # 问题（已回答）：atexit.register(self.exit) 是什么？
        # 回答：它把 exit 注册为解释器正常退出时的清理回调，确保 worker、共享内存、CUDA Graph 和
        # distributed process group 被释放；_exited 让显式调用和 atexit 再次调用时保持幂等。
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        for p in self.ps:
            # 问题（已回答）：Process.join() 是什么意思？
            # 回答：主进程在这里等待 worker 执行完 exit 并终止，然后回收其进程资源；它不是把数据合并起来。
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            raise ValueError("Prompt must contain at least one token")
        # 问题（已回答）：为什么检查时要加 sampling_params.max_tokens？
        # 回答：生成过程会把最多 max_tokens 个 completion token 继续追加到同一序列，因此最坏总长度是二者之和；
        # 只检查 prompt 会允许后续 decode 越过 max_model_len。实际遇到 EOS 时可以提前结束，但不能依赖它一定出现。
        if len(prompt) + sampling_params.max_tokens > self.max_model_len:
            raise ValueError(
                f"Prompt ({len(prompt)} tokens) and requested completion "
                f"({sampling_params.max_tokens} tokens) exceed max_model_len={self.max_model_len}"
            )
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # 问题（已回答）：为什么 prefill 求和，decode 却记录 -len(seqs)？
        # 回答：prefill 中每条序列本轮可处理多个 scheduled token，所以工作量是它们的总和；decode 中每条序列
        # 恰好只处理一个待解码 token，所以 token 数就是序列数。负号只是 generate 用来区分两类吞吐量的内部标记，
        # 不表示真的处理了负数个 token，也不参与模型计算。
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 问题（已回答）：为什么需要后处理，postprocess 做什么？
        # 回答：模型只返回本轮采样 token；调度状态仍需在 CPU 侧更新。postprocess 会登记刚完成的整块 prefix hash、
        # 推进 num_cached_tokens、处理未完成的 chunked prefill、追加有效采样 token，并在 EOS/长度上限时结束请求和回收 KV blocks。
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 问题（已回答）：如果序列尚未完成会怎样？
        # 回答：它不会进入本次 outputs；仍保留在 waiting/running 队列，后续 step 继续 prefill 或 decode。
        # 只有 FINISHED 序列才把完整 completion 交给 generate。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        # 问题（已回答）：list[list[int]] 是否表示多个用户已经转换好的 token id？
        # 回答：它表示一批独立请求的 token-id 序列，不携带用户身份；这些请求可以来自一个用户或多个用户。
        # 多用户在线服务还需要在本 API 外增加并发接入、请求 ID、取消和单一 engine loop，不能仅靠这个类型判断用户。
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        # 问题（已回答）：use_tqdm 的作用是什么？
        # 回答：它只控制是否显示终端进度条和吞吐量，不改变调度、采样或返回结果。
        use_tqdm: bool = True,
    ) -> list[str]:
        # 问题（已回答）：这里为什么创建 tqdm 对象？
        # 回答：进度条的 total 是请求数；每完成一条序列就推进一次，并在 postfix 展示最近一次 prefill/decode 吞吐量。
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 问题（已回答）：一个 SamplingParams 为什么要复制给全部 prompt，何时需要逐请求参数？
        # 回答：传单个对象表示整批共享 temperature、max_tokens 等设置；传列表则可让不同请求采用不同温度、
        # 输出预算或 EOS 策略。这里复制的是对象引用，但 Sequence 立即读取其中的标量字段，因此不会共享运行时状态。
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()

            output, num_tokens = self.step()
            # 问题（已回答）：为什么 num_tokens > 0 表示 prefill？
            # 回答：step 约定用正数记录 prefill 的实际 scheduled token 总数，用负数记录 decode 的序列数；
            # 因此这里是在解释该内部符号约定，而不是从模型输出推断阶段。
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                # 问题（已回答）：decode 吞吐量为什么取 -num_tokens？
                # 回答：decode 的 num_tokens 被故意编码为负数以标记阶段；取反恢复本轮实际处理的 token 数，
                # 即每条 scheduled sequence 各一个 token，才能得到正的 tokens/s。
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            # 问题（已回答）：这里的 token_ids 是 completion_token_ids 吗？
            # 回答：是，只包含该已完成序列生成的 completion，不包含 prompt；未完成序列不会出现在 output 中。
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                # 问题（已回答）：pbar.update(1) 更新什么？
                # 回答：它把“已完成请求数”增加 1，不是增加一个 token；进度条 total 也是 prompts 的数量。
                pbar.update(1)
        pbar.close()

        # 问题（已回答）：outputs 的 key 是否为 seq_id，为什么排序后再返回？
        # 回答：是。请求可能因长度不同而乱序完成，按单调递增的 seq_id 排序可恢复本次加入 scheduler 的输入顺序，
        # 使返回列表继续与 prompts 一一对应。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
