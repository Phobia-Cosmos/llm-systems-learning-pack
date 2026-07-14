# MiniMind 初学者路线图

这份说明按“先能跑通，再理解每一层”的顺序整理当前项目。

## 1. 先认识目录

```text
minimind/
├── model/                 # tokenizer 与 MiniMind 模型结构
├── dataset/               # 训练数据放这里
├── trainer/               # 各阶段训练脚本
├── scripts/               # 导出、WebUI、API 等工具脚本
├── eval_llm.py            # 本地命令行推理入口
├── out/                   # 训练出的轻量权重 .pth
├── checkpoints/           # 可断点续训的完整训练状态
└── artifacts/             # 建议放导出后的 CPU/上传包
```

初学者优先只关注 4 个文件：

- `model/model_minimind.py`：模型长什么样。
- `dataset/lm_dataset.py`：jsonl 数据如何变成 token。
- `trainer/train_pretrain.py`：预训练，学语言接龙。
- `trainer/train_full_sft.py`：监督微调，学对话和指令。

## 2. 项目完整流程

1. 建 Python 环境。
2. 下载 `pretrain_t2t_mini.jsonl` 和 `sft_t2t_mini.jsonl` 到 `dataset/`。
3. 预训练得到 `out/pretrain_512.pth`。
4. 基于预训练权重做 SFT，得到 `out/full_sft_512.pth`。
5. 用 `eval_llm.py` 测试。
6. 导出 Transformers 格式到 `artifacts/minimind-cpu-512/`。
7. 打包或上传导出的模型目录。

## 3. 本机已采用的训练配置

为了后续 CPU 电脑也能运行，本次使用较小配置：

```text
hidden_size        = 512
num_hidden_layers  = 8
params             ≈ 30M
pretrain seq_len   = 340
sft seq_len        = 512
epochs             = 1 + 1
```

这个配置的目标是“完整跑通从 0 到可对话模型”，不是追求榜单效果。若以后更重视效果，可把 `hidden_size` 改回默认 `768`，并增加训练轮数或使用完整数据集。

## 4. 复现命令

当前工作区统一复用 Disk 下的共享 AI 环境：

```bash
cd /home/undefined/Desktop/ai
source /home/undefined/Desktop/ai/use_disk_ai_env.sh
```

当前验证可用环境：

```text
/home/undefined/Disk/ai-storage/.venv-sglang
```

它已经包含 `torch`、`transformers`、`datasets`、`tokenizers`、`safetensors`、`modelscope` 等 MiniMind 需要的核心依赖。不要在 `projects/minimind` 下再创建 `.venv`。

下载数据：

```bash
modelscope download --dataset gongjy/minimind_dataset pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl --local_dir dataset
```

预训练：

```bash
cd trainer
python train_pretrain.py \
  --epochs 1 \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 340 \
  --batch_size 64 \
  --accumulation_steps 1 \
  --num_workers 4 \
  --log_interval 500 \
  --save_interval 20000 \
  --dtype bfloat16 \
  --device cuda:0
```

SFT：

```bash
cd trainer
python train_full_sft.py \
  --epochs 1 \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 512 \
  --batch_size 48 \
  --accumulation_steps 1 \
  --num_workers 4 \
  --log_interval 500 \
  --save_interval 20000 \
  --dtype bfloat16 \
  --device cuda:0
```

测试原生权重：

```bash
cd /home/undefined/Desktop/ai/projects/minimind
python eval_llm.py \
  --load_from model \
  --weight full_sft \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --device cpu \
  --max_new_tokens 256
```

## 5. 导出 CPU 运行包

训练完成后执行：

```bash
cd scripts
python export_cpu_model.py \
  --torch_path ../out/full_sft_512.pth \
  --output_dir ../artifacts/minimind-cpu-512 \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --dtype float32
```

然后可以直接用 Transformers 格式推理：

```bash
cd /home/undefined/Desktop/ai/projects/minimind
python eval_llm.py \
  --load_from artifacts/minimind-cpu-512 \
  --device cpu \
  --max_new_tokens 256
```

## 6. 上传模型参数

没有目标仓库和登录 token 时，不能直接替你上传到 HuggingFace 或 ModelScope。当前项目会先生成本地上传包：

```bash
tar -czf artifacts/minimind-cpu-512.tar.gz -C artifacts minimind-cpu-512
```

如果要上传到 HuggingFace：

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload YOUR_NAME/minimind-cpu-512 artifacts/minimind-cpu-512 .
```

如果要上传到 ModelScope，先登录 ModelScope CLI，再上传 `artifacts/minimind-cpu-512/` 或压缩包。

## 7. 学习顺序

建议按这个顺序看代码：

1. 先看 `dataset/lm_dataset.py` 的 `PretrainDataset` 和 `SFTDataset`。
2. 再看 `trainer/train_pretrain.py` 的 1 到 9 个步骤注释。
3. 接着看 `trainer/train_full_sft.py`，比较它和预训练的差别。
4. 最后看 `model/model_minimind.py`，重点理解 RMSNorm、Attention、FeedForward、Decoder Block。

先不要碰 PPO、GRPO、DPO、Agent、LoRA。等预训练和 SFT 都能稳定跑通，再进入这些进阶阶段。
