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
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = load_tokenizer(config.model, config.hf_config)
        # 问题（已回答）：分词器需要 eos_token_id 吗？
        # 回答：engine 需要统一读取它来判断生成何时结束。没有 EOS 的教学 tokenizer 可返回 None/-1，
        # 调用方再用 ignore_eos=True 或 max_tokens 作为停止条件。
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._exited = False
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        self.model_runner.call("exit")
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            raise ValueError("Prompt must contain at least one token")
        if len(prompt) + sampling_params.max_tokens > self.max_model_len:
            raise ValueError(
                f"Prompt ({len(prompt)} tokens) and requested completion "
                f"({sampling_params.max_tokens} tokens) exceed max_model_len={self.max_model_len}"
            )
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        # TODO：list[list[int]]指的是多用户同时输入转换为token id了吧？
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        # TODO:这个变量的作用是什么？
        use_tqdm: bool = True,
    ) -> list[str]:
        # TODO：这个函数的作用是什么？
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # TODO：如果是多个prompt但是只有一个sampling param那就复制每一个prompt相同的采样参数？
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
