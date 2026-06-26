# 近 5 年 LLM 发展主线与论文地图

时间范围按 2021-2026 理解。2017 年 Transformer 和 2020 年 GPT-3 不在近 5 年内，但它们是理解后续论文的前置基础。

## 主线 1: Scaling law 从“大模型”转向“数据和计算最优”

代表论文：

- GPT-3, Language Models are Few-Shot Learners, NeurIPS 2020: 大模型少样本能力的标志性工作。
- Chinchilla, Training Compute-Optimal Large Language Models, 2022: 说明很多大模型参数很大但训练 token 不够，模型大小和训练 token 都要随计算预算一起扩展。
- LLaMA, Open and Efficient Foundation Language Models, 2023: 证明高质量公开数据和充分训练可以让较小模型接近或超过更大模型。

你需要关注的问题：

- 参数量、数据量、计算量之间怎么权衡。
- 为什么很多开源模型选择“更小但训练更久”。
- 数据质量和去重为什么变得越来越重要。

## 主线 2: 架构从标准 Transformer 走向高效注意力、长上下文和 MoE

代表论文：

- Switch Transformers, JMLR 2022: 稀疏激活 MoE，用更多总参数但每个 token 只激活部分专家。
- FlashAttention, NeurIPS 2022: 不是近似 attention，而是 IO-aware 的精确 attention，大幅减少 GPU HBM 读写。
- Mistral 7B, 2023: GQA 和 sliding window attention，提高推理效率和长上下文能力。
- Mixtral of Experts, 2024: 稀疏 MoE，每个 token 只用部分专家，降低推理计算。
- DeepSeek-V3, 2024: MoE、MLA、多 token prediction、训练系统优化结合。

你需要关注的问题：

- attention 的瓶颈常常是显存带宽和 KV cache。
- 长上下文不是只把 position embedding 拉长，还涉及训练数据、attention 计算、评测和检索能力。
- MoE 的重点是路由、负载均衡、专家 specialization 和通信开销。

## 主线 3: 训练后阶段变得和预训练一样重要

代表论文：

- InstructGPT, Training language models to follow instructions with human feedback, NeurIPS 2022: SFT + reward model + RLHF，使模型更会遵循人类指令。
- Constitutional AI, 2022: 用规则和 AI feedback 减少人工偏好标注压力。
- DPO, Direct Preference Optimization, NeurIPS 2023: 用更简单稳定的分类式目标替代复杂 RLHF 流程。
- QLoRA, NeurIPS 2023: 4-bit 量化基座 + LoRA adapter，使普通单卡也能微调大模型。

你需要关注的问题：

- Base model 和 chat/instruct model 的区别。
- SFT、RLHF、DPO、RLAIF 分别解决什么问题。
- 参数高效微调为什么适合个人和实验室研究。

## 主线 4: 推理能力从 prompt 技巧走向训练目标和 test-time compute

代表论文：

- Chain-of-Thought Prompting, NeurIPS 2022: 大模型通过中间推理步骤提升数学、符号、常识推理。
- Self-Consistency, ICLR 2023: 采样多条推理路径，用一致性投票选答案。
- ReAct, ICLR 2023: 把 reasoning 和 action 交替起来，让模型调用外部环境或工具。
- Tree of Thoughts, NeurIPS 2023: 把生成过程变成搜索过程，探索多个思路。
- DeepSeek-R1, 2025: 用强化学习激发推理行为，并开源 reasoning 模型和蒸馏模型。

你需要关注的问题：

- 推理能力不只是模型参数，还包括训练数据、RL、采样策略和验证器。
- test-time compute 是重要方向：推理时多算一些，可能换来更好答案。
- reasoning model 成本更高，需要考虑延迟、token 数和评估可靠性。

## 主线 5: LLM 从纯文本走向工具、检索、代码和多模态

代表论文：

- RAG, Retrieval-Augmented Generation, NeurIPS 2020: 把参数知识和外部检索结合，改善可更新性和事实性。
- Toolformer, NeurIPS 2023: 让模型学习调用计算器、搜索、翻译等外部工具。
- Flamingo, NeurIPS 2022: 少样本视觉语言模型，连接视觉编码器和语言模型。
- GPT-4 technical report / Sparks of AGI, 2023: 多模态、代码、数学、法律、医学等综合能力成为评价重点。

你需要关注的问题：

- 真实应用通常不是裸 LLM，而是 LLM + RAG + tools + memory + safety。
- 代码模型、数学模型、多模态模型都有专门数据和训练后流程。
- 工具调用让模型从“生成文本”扩展到“执行任务”。

## 结合本项目怎么读论文

这个仓库里的最小模型对应论文中的核心骨架：

- `tokenizer.py`: 对应 tokenizer、BPE、SentencePiece、数据预处理论文。
- `model.py/CausalSelfAttention`: 对应 Transformer、FlashAttention、GQA、MQA、MLA、长上下文论文。
- `model.py/MLP`: 对应 GELU、SwiGLU、MoE、专家路由论文。
- `train.py`: 对应 scaling law、优化器、学习率、checkpoint、数据混合。
- `generate.py`: 对应 temperature、top-k、top-p、beam search、self-consistency、test-time compute。

不要一开始就追 70B 模型。先把这个小模型跑通，再每次只替换一个模块，记录 loss、速度、显存和生成质量变化。
