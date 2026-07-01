#!/usr/bin/env bash
set -euo pipefail

KEYRING_DEB="/tmp/cuda-keyring_1.1-1_all.deb"
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"

if [ ! -f "$KEYRING_DEB" ]; then
  wget -O "$KEYRING_DEB" "$KEYRING_URL"
fi

sudo dpkg -i "$KEYRING_DEB"
sudo apt update
apt-cache policy cuda-toolkit-13-0
sudo apt install -y cuda-toolkit-13-0

cat <<'MSG'

CUDA Toolkit 13.0 install command completed.
Add this to your shell profile if nvcc is not found automatically:

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

Then verify with:

nvcc --version

MSG
