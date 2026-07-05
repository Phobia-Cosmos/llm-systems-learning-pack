from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig

# 问题（已回答）:除了这一种attention 还可以使用哪些？为什么注意力也是nn.Module？n_embd和n_head分别代表什么以及为什么有这样的关系？这里的head代表什么意思？head可以无限增加吗？head_dim代表一个head处理多少的embed吗？
# 回答：这里实现的是 decoder-only GPT 的 causal multi-head self-attention。其他常见 attention 包括 encoder bidirectional self-attention、
# encoder-decoder cross-attention、multi-query/grouped-query attention、sliding-window attention、linear/sparse attention、FlashAttention 等。

# 注意力写成 nn.Module，是因为它有可训练参数 Wq/Wk/Wv/Wo，也需要被 PyTorch 注册、保存、移动设备和参与反向传播。
# n_embd 是每个 token 的隐藏向量总维度；n_head 是把这个总维度切成多少个注意力头；head 表示一组独立的 Q/K/V 子空间。
# n_embd 必须能整除 n_head，因为每个头拿到 head_dim = n_embd // n_head 维。head 不能无限增加：head_dim 太小会损失表达能力，
# 头太多也会增加 kernel/通信/显存开销。是的，head_dim 就是一个 head 处理的 embedding 子维度。
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # 问题（已回答）:c_attn和c_proj是什么？这两个变量的作用是什么在Transformer内？为什么这两个变量内部要如何设置Linear，也要给我解释清楚。
        # 回答：c_attn 是一次性生成 Q、K、V 的线性层，把每个 token 的向量从 n_embd 投影到 3*n_embd，
        # 后面再 split 成 q/k/v。这样等价于三个 Linear，但更紧凑。c_proj 是 attention 输出后的输出投影 Wo，
        # 把多个 head 拼回来的 n_embd 维结果再混合一次，送回残差主干。Linear 的本质是 y = xW^T + b，
        # 这里输入最后一维是 n_embd，所以 in_features=n_embd；QKV 总共三份，所以 out_features=3*n_embd。
        # TODO:为什么可以一次性生成 Q、K、V 的线性层？这里token向量维度是多少？为什么要投影到3倍？为什么这个在论文中没有提到这个概念？紧凑有什么好处吗？多个head返回的结果混合原理是什么？残差主干指的是什么？
        # TODO:请你帮我把网络的结构画出来。proj针对每一个qkv都要做投影还是？
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 问题（已回答）:Dropout是一个函数吗？
        # 回答：nn.Dropout 是一个 Module，不只是普通函数。训练时它按概率随机把部分元素置 0 并缩放剩余元素，
        # 用来防止过拟合；eval() 时它自动关闭，直接返回输入。
        # TODO:为什么要针对atten和residual同时都要做？
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 问题（已回答）:这个是什么？内部生成一个2d矩阵？然后进行什么运算？为什么要有mask？这个mask的作用是什么？register_buffer又是什么？为什么view有四个参数？persistent代表什么意思？
        # 回答：torch.ones 先生成 [block_size, block_size] 的全 1 矩阵，torch.tril 保留下三角，得到 causal mask。
        # 第 i 行只能看第 0..i 列，不能看未来 token；否则训练时模型会偷看答案。

        # register_buffer 注册的是“跟随模型移动设备但不是可训练参数”的张量。view(1,1,T,T) 是为了和 scores 的
        # [batch, head, seq, seq] 广播对齐；前两个 1 分别对应 batch 和 head 维。persistent=False 表示不把这个 mask 存进 state_dict，
        # 因为它可以由 block_size 重新生成，不属于训练得到的权重。
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        # TODO：为什么要生成张量？score是什么？state_dict又是什么？代码中没有出现这个属性呀？
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 问题（已回答）:这里的x是什么 为什么可以直接通过shape赋值三个变量？三个变量分别代表什么意思？
        # 回答：x 是进入 attention 的隐藏状态张量，shape 是 [batch, seq_len, channels]。
        # Python 支持解包赋值，所以 x.shape 这个三元组可以直接拆成 batch、seq_len、channels。
        # batch 是样本数，seq_len 是当前上下文 token 数，channels 通常等于 n_embd。
        batch, seq_len, channels = x.shape
        # 问题（已回答）:为什么可以这样得到qkv？c_attn是在做什么以及为什么要split？为什么要按照channels分？以及为什么dim是2？
        # 回答：self.c_attn(x) 把最后一维从 channels 投影到 3*channels，里面依次放 Q、K、V。
        # split(channels, dim=2) 表示沿最后一维切成三段，每段长度 channels；dim=2 是因为 x 的维度是 [B,T,C]，
        # 第 2 维就是 embedding/channel 维。
        q, k, v = self.c_attn(x).split(channels, dim=2)

        # 问题（已回答）:这里是在做什么 为什么这样做 对应公式的哪一步？为什么要转置？为什么传入四个参数？
        # 回答：这一步把 [B,T,C] 拆成多头形式 [B,T,H,D]，其中 H=n_head、D=head_dim，然后转成 [B,H,T,D]。
        # 这样每个 head 可以独立计算 Attention(Q,K,V)=softmax(QK^T/sqrt(D))V。view 的四个参数就是目标 shape。
        # transpose(1,2) 把 head 维提前，是为了后续矩阵乘法能在每个 batch、每个 head 上并行计算。
        # TODO：转置后这个矩阵变成什么样子了？transpose函数的作用是什么 为什么可以对矩阵内部进行使用？
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # 问题（已回答）:为什么转置还可以是负数？@是什么意思？
        # 回答：负数维度是从后往前数，-1 是最后一维，-2 是倒数第二维；k.transpose(-2,-1) 把 [B,H,T,D] 变成 [B,H,D,T]。
        # @ 是 Python 的矩阵乘法运算符，对张量来说会做 batch matrix multiply。这里计算 QK^T，得到 [B,H,T,T] 的注意力分数。
        # TODO：注意力分数不是Q K V计算出来的吗,为什么会和[B,H,T,T] 的注意力分数相关？
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 问题（已回答）:这里又是在做什么？这个函数的作用是什么？原理？公式是什么？
        # 回答：masked_fill 把 mask 为 True 的位置替换成指定值。这里先找 causal_mask 中为 0 的未来位置，
        # 再把对应 scores 设为 -inf。softmax(-inf)=0，所以未来 token 的注意力权重会变成 0。
        # 数学上是 softmax((QK^T / sqrt(d)) + mask)，其中 mask 的未来位置是 -inf。
        # TODO：masked_fill是nn内置的函数是吗？一定要现有mask才能使用吧？这里的F是什么？
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        # 问题（已回答）:为什么要单独dropout？
        # 回答：attention weights dropout 随机丢掉一部分“看向其他 token 的连接”，是 Transformer 里常见正则化。
        # 后面 resid_dropout 作用在输出投影后，两者位置不同：一个正则化注意力分布，一个正则化残差分支输出。
        weights = self.attn_dropout(weights)
        y = weights @ v

        # 问题（已回答）:contiguous的作用是什么？为什么要变成view？
        # 回答：transpose 后张量的内存步长可能不是连续的，view 要求按连续内存解释 shape。
        # contiguous 会拷贝/整理成连续内存。这里 view 把 [B,T,H,D] 重新拼回 [B,T,C]，让多头结果回到原 embedding 维度。

        # TODO：为什么transpose 后张量的内存步长可能不是连续的？
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        # 问题（已回答）:这里是在计算残差吗？
        # 回答：这里还不是残差相加，只是在计算 attention 分支输出。真正的残差在 TransformerBlock.forward 里：
        # x = x + self.attn(self.ln_1(x))。这里返回的是要被加回主干的那一支。
        # TODO：残差是什么以及为什么需要残差？
        return self.resid_dropout(self.c_proj(y))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        total_len = past_len + seq_len
        if total_len > self.causal_mask.size(-1):
            raise ValueError("KV cache length exceeds block_size; use a longer block_size or fewer generated tokens")

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, past_len:total_len, :total_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.c_proj(y)), (k, v)

# 问题（已回答）:为什么nn中还有Sequential？nn中都包含哪些东西？
# 回答：nn.Sequential 是把多个层按顺序串起来的容器，适合“输入依次经过 A、B、C”的简单网络。
# torch.nn 里包含 Module 基类、Linear/Embedding/Conv、LayerNorm/BatchNorm/RMSNorm 类似归一化层、Dropout、激活函数、损失函数等。
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # 问题（已回答）:为什么1in 4out然后接一个GELU后反过来？这个linear的选择有什么特殊的地方没？只能选择linear吗？GELU的作用是什么？为什么要Dropout？
        # 回答：Transformer 的 MLP 通常先把维度扩到 4*n_embd，再压回 n_embd，给每个 token 更多非线性变换容量。
        # Linear 是最常见选择，现代 LLM 也常用 SwiGLU/GEGLU 这类门控 MLP，不是只能用普通 Linear+GELU。
        # GELU 是平滑激活函数，引入非线性；如果只有线性层叠加，整体仍等价于一个线性变换。
        # Dropout 训练时随机丢一部分激活，降低 toy 语料上的过拟合。
        # TODO：为什么是变为4而不是其他的数量的embed？线性层后面一定要接激活函数吗？为什么后一个linear使用的是Dropout而不是激活函数了？
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    # 问题（已回答）:这个前向传播为什么只是声明一个网络？
    # 回答：网络结构已经在 __init__ 的 self.net 里声明好了；forward 只需要说明数据如何流过它。
    # self.net(x) 会自动按 Sequential 里的顺序执行 Linear -> GELU -> Linear -> Dropout。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 问题（已回答）:ln_1和2代表什么？Transformer的一般结构是什么？MLP的作用是什么？
# 回答：ln_1 和 ln_2 是两个 LayerNorm，分别放在 attention 分支和 MLP 分支前面，这叫 pre-norm 结构。
# 一个 decoder-only Transformer block 通常是：x -> LN -> causal self-attention -> residual add -> LN -> MLP -> residual add。
# attention 负责让 token 从上下文中取信息；MLP 负责对每个 token 位置独立做非线性特征变换。
class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present_kv = self.attn.forward_with_cache(self.ln_1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv

# 问题（已回答）:nn.Module是什么？为什么需要token_embedding和position_embedding？ModuleList是什么？nn中的Norm都有哪些选项？lm_head是什么？
# 回答：nn.Module 是 PyTorch 所有可训练网络模块的基类，负责参数注册、设备迁移、train/eval 模式、state_dict 保存等。
# token_embedding 把离散 token id 变成连续向量；position_embedding 告诉模型 token 在序列中的位置，否则 self-attention 本身不区分顺序。
# ModuleList 是保存一组子模块的列表容器，能让 PyTorch 正确发现里面每个 block 的参数。
# 常见 Norm 包括 LayerNorm、BatchNorm、GroupNorm，现代 LLM 也常用 RMSNorm。lm_head 是语言模型输出头，
# 把隐藏向量映射到 vocab_size 个 logits，用来预测下一个 token。
class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # TODO：各个参数的意义是什么？为什么token和position不一致？
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        # 问题（已回答）:为什么要设置n_embd？ln_f的作用是什么？
        # 回答：LayerNorm 需要知道归一化最后一维的大小，这里最后一维就是 n_embd。
        # ln_f 是所有 Transformer block 之后的最终归一化，帮助输出分布稳定，再交给 lm_head 预测词表 logits。
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # TODO：为什么token embedding会有默认的weight 我们传入了吗？
        self.lm_head.weight = self.token_embedding.weight

        # TODO：这个是在调用谁的函数？那里定义的？nn.Module？
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # 问题（已回答）:为什么初始化全是正态分布？
        # 回答：神经网络不能把权重全初始化成 0，否则不同神经元会学到完全相同的东西。
        # 小方差正态分布是 GPT 系列常见的简单初始化，让初始激活和梯度规模比较稳定。
        # 真实大模型还会根据层数、残差分支做更精细的缩放初始化。
        # TODO：为什么初始化为0就会学到相同的内容？这里初始化的缩放原理是什么？为什么针对不同的层 初始化方式不同？为什么MLP不初始化？为什么激活函数不初始化？
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 问题（已回答）:为什么idx是一个tensor？为什么idx还有device属性？
    # 回答：idx 是一批 token id，模型计算需要批量张量而不是 Python list。Tensor 记录 dtype、shape、device 等信息；
    # device 表示它在 CPU、CUDA GPU 或 MPS 上，后续新建 positions 时要放在同一设备。
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 问题（已回答）:为什么idx有这两个属性？
        # 回答：idx 的 shape 是 [batch, seq_len]：每一行是一条 token 序列。batch 表示一次并行处理多少条序列，
        # seq_len 表示每条序列当前有多少个 token。
        batch, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")

        # 问题（已回答）:为什么要与device关联？为什么token embed和位置embed可以直接相加？block(x)是什么？
        # 回答：positions 必须和 idx/model 在同一 device，否则 embedding 查表和相加会报设备不一致。
        # token_embedding(idx) 的 shape 是 [B,T,C]，position_embedding(positions) 的 shape 是 [T,C]，PyTorch 会广播成 [B,T,C] 后相加。
        # 含义是“这个 token 是什么”加上“它在第几个位置”。block(x) 调用的是 TransformerBlock.forward，依次执行 LN、attention、残差、MLP。
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # 问题（已回答）:logits是token还是什么？
        # 回答：logits 不是 token，而是每个位置对词表中每个 token 的未归一化分数，shape 是 [B,T,vocab_size]。
        # softmax 后才变成概率；取 argmax 或采样后才得到真正的下一个 token id。
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 问题（已回答）:F.cross_entropy在算什么？为什么传入的logits是view？为什么是三个参数？这个loss用来做什么的？
            # 回答：cross_entropy 计算“模型预测的词表分布”和“正确下一个 token id”之间的差距，内部等价于 log_softmax + NLLLoss。
            # logits 原来是 [B,T,V]，targets 是 [B,T]；view(-1,V) 和 view(-1) 把所有 batch/时间位置摊平成 [B*T,V] 与 [B*T]。
            # 这里实际传了两个核心参数：预测 logits 和目标 ids；第三个信息 logits.size(-1) 是给 view 用的词表维度。loss 用来反向传播，
            # 告诉优化器该怎样调整权重，让正确 token 的概率变高。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def forward_with_cache(
        self,
        idx: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len = idx.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(2)
        total_len = past_len + seq_len
        if total_len > self.config.block_size:
            raise ValueError(f"sequence length {total_len} exceeds block_size {self.config.block_size}")

        positions = torch.arange(past_len, total_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.drop(x)

        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, block in enumerate(self.blocks):
            past_kv = None if past_key_values is None else past_key_values[layer_idx]
            x, present_kv = block.forward_with_cache(x, past_kv)
            present_key_values.append(present_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, present_key_values

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        logits = logits / max(temperature, 1e-6)
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
        if greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # 问题（已回答）:这个@会有什么用？decorator的作用是什么？top_k的作用是什么？
    # 回答：@ 是装饰器语法，把下面的 generate 函数交给 torch.no_grad() 包装。生成阶段不训练，不需要保存梯度，
    # 所以 no_grad 可以省显存、加快速度。top_k 是采样截断：只允许从分数最高的 k 个 token 中采样，减少低质量长尾 token。
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # 问题（已回答）:idx_cond得到的是什么？为什么后续还可以直接self？logits的各个维度分别代表什么？
            # 回答：idx_cond 是当前上下文的最后 block_size 个 token，因为模型最多只能处理 block_size 长度。
            # self(idx_cond) 等价于调用 self.forward(idx_cond)，这是 nn.Module 的 __call__ 机制。
            # logits 的 shape 是 [batch, seq_len, vocab_size]；这里随后取 logits[:, -1, :]，只用最后一个位置预测下一个 token。
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # 问题（已回答）:为什么dim=-1？probs代表什么意思？一个token？为什么idx next是multinomial计算出来的？这个函数主要在做什么？还有最后的cat又在做什么？
            # 回答：dim=-1 表示在最后一维 vocab_size 上做 softmax，让每行词表分数变成概率分布。probs 是“下一个 token 是每个词表项的概率”，
            # 不是单个 token。torch.multinomial 按概率分布随机抽样，返回采到的 token id，shape [B,1]。
            # torch.cat((idx, idx_next), dim=1) 把新 token 接到序列末尾，下一轮继续用更长上下文生成。
            idx_next = self._sample_next_token(logits, temperature=temperature, top_k=top_k, greedy=greedy)
            idx = torch.cat((idx, idx_next), dim=1)
        # 问题（已回答）:返回的idx代表什么意思？是我们人类能理解的文本吗？
        # 回答：返回的 idx 是完整 token id 序列，包括原 prompt 和新生成 token。它不是人类可读文本；
        # 需要交给 tokenizer.decode(idx.tolist()) 才能变回字符串。
        return idx

    @torch.no_grad()
    def generate_with_kv_cache(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        if idx.size(1) + max_new_tokens > self.config.block_size:
            raise ValueError(
                "MiniLLM uses learned absolute position embeddings, so this teaching KV-cache path "
                "requires prompt_len + max_new_tokens <= block_size."
            )

        logits, past_key_values = self.forward_with_cache(idx)
        for _ in range(max_new_tokens):
            idx_next = self._sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                greedy=greedy,
            )
            idx = torch.cat((idx, idx_next), dim=1)
            logits, past_key_values = self.forward_with_cache(idx_next, past_key_values)
        return idx

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
