# MiniLLM 160M 8K continuation

This directory is the local copy of the four-A100 8K continuation artifacts produced on 2026-07-31. The authoritative checkpoint remains on the server at `/public/home/u43077/lzh/outputs/minillm-general/new-model-v1/160m-openbpe-32k-8k/latest.pt`.

The run resumed the completed 4K checkpoint at step 211345 and 10,000,007,168 cumulative tokens. It used four A100-SXM4-40GB GPUs, sequence length 8192, micro-batch 4 per rank, accumulation 1, and 131,072 global tokens per optimizer update. The final checkpoint is step 218975 with 11,000,086,528 cumulative tokens.

| Added 8K tokens | 8K loss | Tail loss | 4K regression loss | Passkey teacher-forced accuracy | Static/dynamic cache parity |
|---:|---:|---:|---:|---:|:---:|
| 0 | 2.7043 | 3.4047 | 2.5794 | 0.639 | pass |
| 100,007,936 | 2.3869 | 2.5330 | 2.5823 | 0.667 | pass |
| 250,085,376 | 2.3801 | 2.5252 | 2.5841 | 0.667 | pass |
| 500,039,680 | 2.3707 | 2.5173 | 2.5823 | 0.667 | pass |
| 1,000,079,360 | 2.3560 | 2.5034 | 2.5741 | 0.667 | pass |

`capacity-summary.json` contains the three capacity candidates. `8k-stage-plan.json` records hashes and token-aligned milestones. `benchmarks/long-context-summary.json` and `.md` are the generated trajectory summary; the two benchmark subdirectories retain each milestone's detailed results. `8k-train.log` and `8k-controller.log` retain the execution logs. Checkpoints are intentionally not copied locally.
