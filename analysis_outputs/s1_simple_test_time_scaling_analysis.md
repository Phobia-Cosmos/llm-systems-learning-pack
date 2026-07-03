# Paper Analysis

论文：s1: Simple test-time scaling

作者：Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, Tatsunori Hashimoto

会议/期刊：EMNLP 2025 正式发表版本；本地 PDF/arXiv v3 为 2025-03-01 版本。注：本地 PDF 元数据里有 "Proceedings of ICML 2025"，但 ACL Anthology 正式元数据给出的 venue 是 EMNLP 2025。

年份：2025

源码/项目：

- GitHub: https://github.com/simplescaling/s1
- arXiv: https://arxiv.org/abs/2501.19393
- ACL Anthology: https://aclanthology.org/2025.emnlp-main.1025/
- s1-32B: https://huggingface.co/simplescaling/s1-32B
- s1.1-32B: https://huggingface.co/simplescaling/s1.1-32B
- s1K: https://huggingface.co/datasets/simplescaling/s1K
- s1K-1.1: https://huggingface.co/datasets/simplescaling/s1K-1.1

本报告主要依据本地 PDF `/home/undefined/Desktop/ai/papers/02_pretrain_sft_alignment/2025ACL-s1 Simple Test-Time Scaling.pdf`，并轻量核对官方论文页、代码仓库和 Hugging Face 页面。网页状态核对日期：2026-07-02。

论文速览：

| Item | Content |
| --- | --- |
| 研究对象 | 推理型大语言模型的 test-time scaling，也就是通过推理时额外计算提升性能 |
| 核心问题 | 在不使用复杂 RL、大规模数据或闭源流程的情况下，能否复现类似 o1 的推理时扩展曲线 |
| 方法摘要 | 用 1,000 条高质量/高难度/多样化推理样本对 Qwen2.5-32B-Instruct 做 SFT，再用 budget forcing 在解码时控制思考 token 预算 |
| 模型 | s1-32B，基座为 Qwen2.5-32B-Instruct |
| 数据 | s1K: 从 59,029 个候选问题中筛到 1,000 个问题，每题配 Gemini Flash Thinking 生成的 reasoning trace 和答案 |
| 后续版本 | s1.1: 复用同一批问题，但用 DeepSeek-R1 重新生成 reasoning traces |
| 任务 | 数学竞赛、科学推理、长链路推理 |
| 评测集 | AIME24, MATH500, GPQA Diamond |
| 主指标 | Accuracy/pass@1；论文另定义 Control、Scaling、Performance 衡量推理时扩展方法 |
| 训练成本 | s1K SFT 约 26 分钟，16 张 H100，即约 7 H100-hour；59K-full 消融约 394 H100-hour |
| 开放性 | 代码、模型权重、数据均开放；这是本文相对于 o1 类闭源系统的重要贡献 |

一句话总结：这篇论文提出了一个极简但很有启发性的 recipe - "小而精的推理蒸馏数据 + 推理时强制延长/截断思考"。它的价值不在于发明复杂训练算法，而在于证明：对一个已经具备潜在推理能力的强基座模型，少量精心设计的 SFT 样本也可能激活长推理行为，并且这种行为可以用简单解码干预获得一定的 test-time scaling。

## 第一部分 研究背景

过去几年语言模型能力提升主要依赖 train-time scaling：更大模型、更大语料、更长训练、更优算力配置。Kaplan scaling laws 和 Chinchilla-style compute-optimal training 之后，行业形成了清晰的训练侧扩展路径。但 o1 类模型出现后，新的问题变成：模型在推理阶段能不能通过花更多计算来变强？

OpenAI o1 展示了强推理能力和随 inference compute 增长而提升的现象，但没有公开训练数据、训练算法和推理控制细节。随后出现了大量复现路线：MCTS/tree search、多 agent、process reward model、RL、大规模蒸馏等。DeepSeek-R1 进一步证明开源权重模型可以达到很强推理性能，但它仍依赖大规模 RL/多阶段训练。

本文的动机是把问题压到最小：如果只想得到一个开放、可复现、能展示 test-time scaling 曲线的推理模型，最少需要什么？作者的答案是：

1. 不一定先做复杂 RL，先看高质量 SFT 是否足够。
2. 不一定需要几十万/百万条推理轨迹，关键是样本选择。
3. 不一定需要模型自己可靠遵守 "think for N tokens"，可以在解码器外部直接控制停止 token。

这使论文更像一篇 "minimal reproducible baseline"。它把 o1 类系统中不透明的部分拆成两个可实验的问题：数据选择能否激活推理行为，解码时预算控制能否产生可观测的扩展曲线。

## 第二部分 历史发展

| 时间 | 代表工作/事件 | 与本文关系 |
| --- | --- | --- |
| 2020-2022 | Kaplan scaling laws, Chinchilla | 主流能力提升来自训练侧 compute scaling |
| 2022-2023 | Chain-of-Thought, self-consistency, process reward model, LIMA | 长推理提示、验证器和少样本对齐为本文提供背景 |
| 2024 | OpenAI o1 | 展示 test-time compute scaling，但方法闭源 |
| 2024 | Qwen2.5, Gemini Flash Thinking, QwQ, Sky-T1/Bespoke 等 | 本文使用 Qwen2.5-32B-Instruct 作为基座，用 Gemini Thinking 生成蒸馏轨迹，并与多个开放推理模型比较 |
| 2025-01 | DeepSeek-R1, s1 arXiv 初稿 | DeepSeek-R1 展示大规模 RL 路线；s1 展示 1K SFT + budget forcing 的极简路线 |
| 2025-02 | s1.1, LIMO | s1.1 用 DeepSeek-R1 traces 替代 Gemini traces；LIMO 进一步支持 "少量高质量样本可激活推理" 的方向 |
| 2025-11 | EMNLP 2025 正式发表 | ACL Anthology 将本文收录为 EMNLP 2025 main paper |

本文在历史脉络中的定位：不是替代 DeepSeek-R1/o1 的完整训练范式，而是把 "强推理模型必须依赖海量后训练数据和复杂 RL" 这个假设往回拉了一步，证明简单 SFT 和解码控制已经能解释相当一部分现象。

## 第三部分 本文创新

### 核心想法

作者认为，Qwen2.5-32B-Instruct 这类强基座模型在预训练中已经见过大量数学、科学、代码和推理模式，推理能力可能已经潜伏在参数中。后训练阶段不一定要从零学习推理，而是提供少量 "认知模板" 来激活长链路推理格式。

具体做法：

1. 构建 s1K：从 59K 候选问题中筛出 1K 个难、多样、高质量的问题，每题配 Gemini Flash Thinking 生成的 reasoning trace。
2. SFT：在 Qwen2.5-32B-Instruct 上训练这些 traces，得到 s1-32B。
3. Budget forcing：推理时通过截断或延长思考阶段控制 compute budget。
4. 用 AIME24、MATH500、GPQA Diamond 验证性能和 test-time scaling 曲线。

### 主要贡献

| 贡献 | 说明 | 重要性 |
| --- | --- | --- |
| s1K 数据构造流程 | 用 Quality、Difficulty、Diversity 三个原则筛 1K 样本 | 证明数据选择比单纯数据量更关键 |
| budget forcing | 解码时强制结束思考，或阻止结束并追加 "Wait" | 给出非常简单、可复现的 test-time compute 控制方法 |
| s1-32B 模型 | 1K SFT 样本就能在数学任务上接近/超过 o1-preview 的部分结果 | 展示样本效率和开放复现价值 |
| test-time scaling 指标 | Control、Scaling、Performance | 不只看最高分，也看能否控制计算量和是否随计算增加而提升 |
| 全开放工件 | 代码、数据、模型权重开放 | 便于社区复现和作为后续研究基线 |

## 第四部分 方法详解

### 5.1 数据构造：从 59K 到 s1K

初始数据池包含 59,029 个问题，来自 16 类来源，包括 NuminaMATH、MATH、OlympicArena、OmniMath、AGIEval、AIME 历史题、TheoremQA、USACO、JEEBench、GPQA、SciEval、s1-prob、LiveCodeBench、s1-teasers 等。

筛选流程：

1. 质量过滤：去掉 API 错误样本，从 59,029 降到 54,116；再去掉格式问题样本，从 54,116 降到 51,581。
2. 难度过滤：用 Qwen2.5-7B-Instruct 和 Qwen2.5-32B-Instruct 解题；若任一模型能正确解决，则认为题目可能偏易并移除。正确性由 Claude 3.5 Sonnet 按参考答案判定。
3. 多样性过滤：用 Claude 3.5 Sonnet 按 MSC-like domain 分类，跨领域均匀采样，同时偏向 reasoning trace 更长的问题。
4. 去污染：对 AIME24、MATH500、GPQA Diamond 做 8-gram overlap 过滤。
5. 最终 s1K：1,000 个问题，覆盖 51 个领域，总 token 约 4.7M。

一个重要但容易忽略的点：最终 s1K 并不是全对数据。论文称 grader 判断 s1K 中约 53.6% 的样本是正确的；s1K-1.1 提升到约 63.0%。作者允许部分错误轨迹存在，因为他们更关注长推理过程的模式学习。这也意味着 s1 的成功更像 "激活推理格式和搜索行为"，而不是单纯从高正确率监督数据中学习答案。

### 5.2 训练

训练配置：

| 项 | 设置 |
| --- | --- |
| 基座 | Qwen2.5-32B-Instruct |
| 目标模型 | s1-32B |
| 数据 | s1K |
| 训练目标 | 标准 next-token prediction / SFT |
| loss | 不在 question 上算 loss，只在 reasoning trace 和 solution 上算 loss |
| 格式 | `<\|im_start\|>think` 与 `<\|im_start\|>answer` 分隔思考和回答 |
| epoch | 5 |
| batch size | 16 |
| steps | 315 |
| precision | bfloat16 |
| LR | 1e-5，5% warmup 后 cosine decay |
| optimizer | AdamW, beta1=0.9, beta2=0.95, weight decay=1e-4 |
| sequence length | 32768 |
| 成本 | 16 H100 上 26 分钟 |

sequence length 是关键超参。短 sequence length 会截断答案段，使模型更多学到 "一直思考" 而不是 "思考后回答"，导致测试时思考 token 变多但性能更差。论文的消融显示，32768 长度比 4096 长度显著更好。

### 5.3 Budget forcing

模型生成被分成 thinking stage 和 answer stage。Budget forcing 的关键是操纵 thinking stage 的结束。

两种控制：

1. 强制减少计算：如果 thinking tokens 达到预算上限，就直接追加 end-of-thinking delimiter，让模型转入 answer stage。
2. 强制增加计算：如果模型试图结束 thinking stage，就屏蔽该结束 token，并把 "Wait" 追加到当前 reasoning trace 后，让模型继续反思或换路尝试。

直观例子是论文里的 raspberry 例子：模型先说 raspberry 有 2 个 r；被追加 "Wait" 后重新检查，发现有 3 个 r。这个例子说明 budget forcing 不是让模型随机变长，而是有机会诱导自我检查。

### 5.4 test-time scaling 指标

论文不只看准确率，还定义了三个指标：

| 指标 | 含义 |
| --- | --- |
| Control | 方法是否能把 thinking compute 控制在目标区间内 |
| Scaling | accuracy 随 compute 增长的平均斜率 |
| Performance | 不同 compute 设置中的最高 accuracy |

这个设计很重要，因为一个方法可能最高分不错，但计算量不可控；也可能计算量变大但准确率不升反降。本文认为好的 test-time scaling 方法应该同时具备可控性、正斜率和高性能。

## 第五部分 实验

| Item | Content |
| --- | --- |
| Datasets | AIME24, MATH500, GPQA Diamond |
| Baselines | Qwen2.5-32B-Instruct, QwQ-32B, o1-preview/o1-mini/o1, Gemini 2.0 Flash Thinking, DeepSeek-R1, DeepSeek-R1-Distill, Sky-T1, Bespoke-32B |
| Metrics | Accuracy/pass@1；Control、Scaling、Performance |
| Ablations | 数据选择、数据量、sequence length、token/step/class conditional control、rejection sampling、budget forcing 追加字符串 |
| Main Results | s1-32B 在数学任务上以 1K 样本达到很强样本效率；budget forcing 使 AIME24 从 50.0 提升到最高约 56.7 |

### 6.1 主结果

论文 Table 1 主要结果如下：

| Model | # reasoning examples | AIME24 | MATH500 | GPQA Diamond |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-32B-Instruct | N/A | 26.7 | 84.0 | 49.0 |
| o1-preview | N/A | 44.6 | 85.5 | 73.3 |
| o1-mini | N/A | 70.0 | 90.0 | 60.0 |
| o1 | N/A | 74.4 | 94.8 | 77.3 |
| QwQ-32B | N/A | 50.0 | 90.6 | 54.5 |
| DeepSeek-R1 | >>800K | 79.8 | 97.3 | 71.5 |
| DeepSeek-R1-Distill | 800K | 72.6 | 94.3 | 62.1 |
| Sky-T1 | 17K | 43.3 | 82.4 | 56.8 |
| Bespoke-32B | 17K | 63.3 | 93.0 | 58.1 |
| s1 w/o BF | 1K | 50.0 | 92.6 | 56.6 |
| s1-32B with BF | 1K | 56.7 | 93.0 | 59.6 |

结论要分开看：

- 相比基座 Qwen2.5-32B-Instruct，s1-32B 有大幅提升，尤其 AIME24 从 26.7 到 50.0/56.7。
- 相比 o1-preview，s1-32B 在 AIME24 和 MATH500 上更强，但在 GPQA Diamond 上明显更弱。
- 相比 o1、DeepSeek-R1、DeepSeek-R1-Distill，s1-32B 不是总体最强；它的优势是样本效率和开放性。
- 相比 Sky-T1/Bespoke 等 open-data 模型，s1-32B 用更少样本达到了相近或更好的部分结果。

### 6.2 数据消融

| 数据版本 | AIME24 | MATH500 | GPQA Diamond | 解释 |
| --- | ---: | ---: | ---: | --- |
| 1K-random | 36.7 | 90.6 | 52.0 | 只重质量，不显式重难度/多样性 |
| 1K-diverse | 26.7 | 91.2 | 54.6 | 只重多样性，不重难度 |
| 1K-longest | 33.3 | 90.4 | 59.6 | 只用长 reasoning trace 作为难度 proxy |
| 59K-full | 53.3 | 92.8 | 58.1 | 样本更多但成本高很多 |
| s1K | 50.0 | 93.0 | 57.6 | 三个标准联合筛选 |

关键结论：更大数据不一定更优。59K-full 没有明显超过 s1K，却需要约 394 H100-hour；s1K 只需约 7 H100-hour。这支持了作者的 sample-efficient reasoning 主张。

### 6.3 test-time scaling 方法消融

| Method | Control | Scaling | Performance on AIME24 | 结论 |
| --- | ---: | ---: | ---: | --- |
| Budget forcing | 100% | 15 | 56.7 | 最终采用的方法 |
| Token-conditional control | 40% | -24 | 40.0 | 模型不能可靠按 token 数停止 |
| Token-conditional + BF | 100% | 13 | 40.0 | 控制改善，但性能不够 |
| Step-conditional control | 60% | 3 | 36.7 | step 数也会被模型绕开 |
| Step-conditional + BF | 100% | 6 | 36.7 | 可控但性能不强 |
| Class-conditional control | 50% | 25 | 36.7 | 粗粒度可行，精细控制弱 |
| Rejection sampling | 100% | -35 | 40.0 | 出现反向 scaling |

budget forcing 的优势在于简单、可控、性能最高。但它也有天花板：多次阻止结束后模型可能进入重复循环，性能曲线会 flatten。

### 6.4 "Wait" 字符串消融

| 设置 | AIME24 | MATH500 | GPQA Diamond |
| --- | ---: | ---: | ---: |
| No extrapolation | 50.0 | 93.0 | 57.6 |
| 2x without string | 50.0 | 90.2 | 55.1 |
| 2x "Alternatively" | 50.0 | 92.2 | 59.6 |
| 2x "Hmm" | 50.0 | 93.0 | 59.6 |
| 2x "Wait" | 53.3 | 93.0 | 59.6 |

"Wait" 并非魔法词，但在作者实验中是最稳定的触发继续检查的字符串。

### 6.5 s1.1

附录 A 说明，s1 发布 7 天后作者发布 s1.1：复用 s1K 的 1,000 个问题，但把 reasoning traces 换成 DeepSeek-R1 生成。官方 GitHub README 也标注 s1.1 是推荐 successor。

本地 PDF Table 5 中，s1.1 相比 s1 明显改善：

| Model | MATH500 | GPQA | AIME 2024 | AIME 2025 |
| --- | ---: | ---: | ---: | ---: |
| s1 w/o BF | 92.6 | 56.6 | 50.0 | 26.7 |
| s1 with BF "Wait" 4x | 92.2 | 58.6 | 56.7 | 36.7 |
| s1.1 w/o BF | 94.4 | 60.6 | 56.7 | 50.0 |
| s1.1 with BF "Wait" 2x | 95.4 | 63.6 | 56.7 | 50.0 |

Hugging Face 当前模型卡还给出了使用 budget forcing 后的 s1/s1.1 对比，并推荐 s1.1。也就是说，若做复现或应用，优先从 s1.1 而不是 s1-32B 原版开始。

## 第六部分 与已有工作的比较

| Work | Key Idea | Difference From This Paper | Limitation |
| --- | --- | --- | --- |
| OpenAI o1 | 通过推理时计算显著提升 reasoning | 本文试图开放复现类似 scaling 行为 | o1 方法、数据、权重闭源 |
| DeepSeek-R1 | 大规模 RL 激励推理能力 | s1 不做大规模 RL，只做 1K SFT + 解码控制 | s1 性能不如 R1/R1-distill，总体能力较弱 |
| QwQ-32B | 开放权重推理模型 | s1 明确开放数据、代码和 budget forcing 方法 | QwQ 训练细节不完全公开 |
| Sky-T1 / Bespoke-Stratos | 用较多蒸馏数据训练开放推理模型 | s1 样本数更少，强调 sample efficiency | s1 部分 benchmark 不占优 |
| LIMA | 1,000 个高质量样本可做对齐 | s1 将类似少样本激活思想扩展到 reasoning | LIMA 不是 test-time scaling 论文 |
| LIMO | 少量高质量数学 reasoning 样本激活推理 | 与 s1 方向一致，进一步强化 "Less is more" 假设 | 更偏数学 reasoning，不一定解决通用推理控制 |
| MCTS / tree search / REBASE | 通过搜索、奖励模型或 verifier 扩展推理时计算 | s1 的 budget forcing 更轻量，不需要外部 verifier | budget forcing 的长期扩展能力较弱 |

我的判断：s1 的真正贡献不是 "打败所有推理模型"，而是给出了一个强基线，迫使后续工作回答：复杂 RL 或 tree search 相比 1K SFT + 简单解码控制，究竟带来了多少额外收益？

## 第七部分 局限性

作者明确提到或实验中暴露的限制：

1. Budget forcing 会遇到循环和上下文窗口上限。多次阻止结束后，模型可能重复、绕圈或耗尽 context window，扩展曲线最终 flatten。
2. 评测非确定性明显。论文附录 B 指出，长推理在 vLLM/greedy 下也可能因 batch size、继续生成、tensor parallelism 等因素导致分数波动。
3. sequence length 对训练很敏感。短上下文会截断答案段，使模型学到过长思考但不稳定回答。

作者没有充分展开、但对后续研究重要的限制：

1. 依赖闭源/强教师模型蒸馏。s1 原版 traces 来自 Gemini Flash Thinking，s1.1 来自 DeepSeek-R1。论文开放了产物，但训练信号仍来自强教师模型。
2. s1K 正确率并不高。最终 s1K 只有约 53.6% 被 grader 判为正确，s1K-1.1 约 63.0%。这说明模型可能学到的是长推理行为模板，而非严格正确推理。
3. GPQA 上不强。s1-32B with BF 在 GPQA Diamond 为 59.6，低于 o1-preview 的 73.3 和 o1 的 77.3，说明科学专业知识/严谨推理并未被同等提升。
4. AIME24 很小。AIME24 只有 30 道题，单题影响 3.33 个百分点，结果容易受评测方差影响。
5. 解码控制较手工。追加 "Wait" 是有效 heuristic，但不是理论上稳健的搜索或验证机制。
6. 复现实验仍需较高资源。虽然 7 H100-hour 相对便宜，但 32B 模型训练和长上下文推理仍不是低门槛实验。
7. 去污染方法有限。8-gram overlap 能过滤显式重叠，但不能完全排除语义等价题或预训练污染。
8. 结论依赖强基座模型。作者假设预训练中已有潜在推理能力；换成弱基座、小模型或低质量 instruct model，1K SFT 未必有效。

未来问题：

- 如何判断模型是在有效反思，还是只是被迫输出更长文本？
- 如何将 budget forcing 从固定字符串 heuristic 升级为带 verifier 的自适应控制？
- 少样本 SFT 激活出来的能力边界在哪里：数学有效，科学/代码/规划是否同样有效？

## 第八部分 最新发展

截至 2026-07-02 核对：

- arXiv 页面显示该论文 2025-01-31 提交，2025-03-01 更新到 v3。
- ACL Anthology 显示正式发表在 EMNLP 2025，页码 20275-20321，DOI 为 `10.18653/v1/2025.emnlp-main.1025`。
- 官方 GitHub 仓库仍可访问，包含 `eval/`、`data/`、`train/` 等目录，并在 README 中推荐 s1.1。
- Hugging Face 上有 s1-32B、s1.1-32B、s1K、s1K-1.1。s1-32B 模型卡明确推荐使用 successor s1.1。
- LIMO 是紧随其后的相关工作，进一步提出少量高质量样本可激活复杂数学推理的观点，并报告 817 个训练样本即可取得强数学结果。

当前研究状态可以概括为三条路线：

1. 少样本 SFT 激活路线：s1、s1.1、LIMO 代表。关注数据质量和认知模板。
2. RL 推理路线：DeepSeek-R1、Kimi k1.5 等代表。关注奖励设计、长链路自我探索和大规模后训练。
3. 推理时搜索路线：majority voting、Best-of-N、REBASE、tree search、process reward model。关注如何在测试时更有效地花 compute。

s1 的位置是第一条路线中最简洁、最开放的代表之一，也是比较第二/第三路线增益时的强基线。

## 第九部分 科研价值

### 研究价值

1. 降低研究门槛。它证明研究 test-time scaling 不一定从大规模 RL 开始，可以先用小数据和解码控制建立基线。
2. 强化数据选择的重要性。1K-random、1K-diverse、1K-longest 都不如 s1K，说明难度、多样性、质量必须联合考虑。
3. 提供可解释的控制指标。Control/Scaling/Performance 比单纯 accuracy 更适合研究推理时扩展。
4. 暴露长推理评测问题。附录 B 对评测非确定性的讨论很实用，提醒后续工作不要过度解读小幅分差。
5. 给 "能力激活" 假说提供证据。强基座模型可能已经具备推理组件，少量样本主要教它何时、以何种格式调用这些能力。

### 可继续做的方向

| 方向 | 具体想法 | 为什么值得做 |
| --- | --- | --- |
| 更强数据选择 | 用 verifier/embedding/uncertainty 联合筛样本，避免只用 trace length 作为难度 proxy | trace length 可能混入低效推理，不等于真实难度 |
| 教师模型比较 | Gemini、DeepSeek-R1、Claude、o-series-like traces 的系统对比 | s1.1 说明教师质量很关键 |
| 自适应 budget forcing | 根据题目难度、当前置信度、重复度动态决定是否追加 "Wait" | 固定 1x/2x/4x 容易浪费 compute 或进入循环 |
| budget forcing + verifier | 追加 "Wait" 后用过程/结果 verifier 判断是否继续 | 让延长思考从 heuristic 变成受控搜索 |
| RL + budget forcing | 在 RL 训练的 reasoning model 上测试同样控制 | 论文也提出这个方向，可能突破 flattening |
| 更稳健评测 | 多 seed、多 batch、不同推理框架、置信区间报告 | 长推理非确定性会影响结论 |
| 跨领域泛化 | 法律、医学、代码调试、规划任务 | 当前主要集中在数学/科学问答 |
| 成本归一化指标 | accuracy per token、accuracy per dollar、Pareto frontier | test-time scaling 的实际价值取决于成本 |

我的优先建议：如果要复现或扩展这篇论文，先不要急着做 RL。更高收益的第一步是复现 s1.1，并加入 verifier-guided adaptive budget forcing。这样能直接检验 "继续想" 是否真的在改正错误，而不是只是拉长输出。

## 第十部分 Roadmap

| Order | Paper/Topic | Why Read It |
| ---: | --- | --- |
| 1 | Chain-of-Thought prompting | 理解长推理轨迹为什么能帮助 LLM |
| 2 | LIMA: Less Is More for Alignment | 理解少量高质量样本激活行为模式的假设 |
| 3 | OpenAI "Learning to reason with LLMs" | 了解 o1 引发 test-time scaling 研究的背景 |
| 4 | DeepSeek-R1 report | 对比大规模 RL 路线与 s1 的极简 SFT 路线 |
| 5 | s1: Simple test-time scaling | 重点读数据构造、budget forcing、Table 1-4、附录 A/B |
| 6 | LIMO: Less is More for Reasoning | 继续理解少样本 reasoning SFT 的后续发展 |
| 7 | REBASE / process reward model / Best-of-N papers | 研究如何把 budget forcing 与更系统的推理时搜索结合 |

建议阅读本文时按这个顺序：

1. 先读 Introduction，抓住作者的问题定义：最简单的 test-time scaling recipe。
2. 再读 Section 2，看 s1K 如何从 59K 中筛出 1K。
3. 接着读 Section 3，理解 budget forcing 和 Control/Scaling/Performance。
4. 重点看 Table 1-4，区分 "性能强"、"样本效率高"、"可控 scaling" 这三件事。
5. 最后读 Appendix A/B/D/E，因为 s1.1、评测非确定性、sequence length、control 方法失败细节都在附录里。

### 知识树

```text
Test-time scaling
├── Sequential scaling
│   ├── Long reasoning traces
│   ├── Budget forcing
│   └── Adaptive stopping / reflection
├── Parallel scaling
│   ├── Majority voting
│   ├── Best-of-N
│   └── Tree search / REBASE
├── Reasoning data
│   ├── Distillation from strong teachers
│   ├── Data quality / difficulty / diversity
│   └── Small-data activation hypothesis
└── Evaluation
    ├── AIME / MATH / GPQA
    ├── pass@1 vs compute-normalized metrics
    └── long-generation determinism
```

### 时间线

| 时间 | 主题 | 推荐关注点 |
| --- | --- | --- |
| 2022-2023 | CoT、self-consistency、LIMA | 长推理和少样本行为激活 |
| 2024 | o1、QwQ、Sky-T1/Bespoke | 推理时扩展成为核心问题 |
| 2025-01 | DeepSeek-R1 与 s1 | 大规模 RL 路线 vs 极简 SFT 路线 |
| 2025-02 | s1.1、LIMO | 教师轨迹质量和少样本 reasoning SFT |
| 后续 | verifier-guided scaling、adaptive compute | 如何更有效、更可控地花推理时计算 |

### 后续论文/主题

- DeepSeek-R1：理解 RL 如何系统性激励长推理。
- LIMO：理解少量高质量 reasoning 样本的边界。
- REBASE / process reward model：理解如何用 verifier 或 reward model 引导搜索。
- Inference scaling laws：理解 compute-budget 与 accuracy 的成本收益关系。
- Evaluation determinism：研究长推理输出在不同推理框架和 batch 设置下的稳定性。

## Sources

- 本地 PDF：`/home/undefined/Desktop/ai/papers/02_pretrain_sft_alignment/2025ACL-s1 Simple Test-Time Scaling.pdf`
- arXiv: https://arxiv.org/abs/2501.19393
- ACL Anthology: https://aclanthology.org/2025.emnlp-main.1025/
- GitHub: https://github.com/simplescaling/s1
- Hugging Face s1-32B: https://huggingface.co/simplescaling/s1-32B
- Hugging Face s1.1-32B: https://huggingface.co/simplescaling/s1.1-32B
- Hugging Face s1K: https://huggingface.co/datasets/simplescaling/s1K
- Hugging Face s1K-1.1: https://huggingface.co/datasets/simplescaling/s1K-1.1
- LIMO: https://arxiv.org/abs/2502.03387
