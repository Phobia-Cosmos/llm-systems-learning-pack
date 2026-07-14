# Current Environment RMSNorm Verification

This repository originally targets Python 2-era experiment stacks for the
full paper reproduction. The current workspace uses Python 3.12, so the
modern verification path is PyTorch-based.

## What Was Modernized

- `rmsnorm_torch.py` now uses the paper-style formula:

```text
x * rsqrt(mean(x^2) + eps)
```

- `p=0` is rejected because partial RMSNorm needs at least one sampled hidden
  dimension.
- `experiments/verify_rmsnorm_paper.py` verifies the paper's core idea without
  TensorFlow 1.x, Theano, Lasagne, or Python 2.

## Run

From the workspace, reuse the shared Disk AI environment:

```bash
cd /home/undefined/Desktop/ai
source /home/undefined/Desktop/ai/use_disk_ai_env.sh
python projects/rmsnorm/experiments/verify_rmsnorm_paper.py \
  --steps 160 --depth 24 --hidden-size 128 --threads 1
```

Verified shared environment:

```text
/home/undefined/Disk/ai-storage/.venv-sglang
```

Do not create or use `projects/rmsnorm/.venv`; this project only needs the modern PyTorch verification path in the shared environment.

## Expected Checks

The script prints:

- `invariance_check`: RMSNorm keeps rescaling invariance and drops shift
  invariance, matching the paper.
- `training_check`: compares `none`, `LayerNorm`, `RMSNorm`, and `pRMSNorm` on
  the same synthetic deep residual MLP classification task.

This is a current-environment validation of the paper's hypothesis, not a
strict reproduction of the WMT/CNN-DM/COCO/CIFAR tables.
