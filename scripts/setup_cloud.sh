#!/usr/bin/env bash
# Cloud GPU environment setup — auto-detects CUDA version, supports sm_120 (Blackwell).
# Run once after each pod restart:
#   bash scripts/setup_cloud.sh

set -e

echo "=== [0/6] Installing system tools (zip, unzip, gdrive, rclone) ==="
if ! command -v zip &>/dev/null || ! command -v unzip &>/dev/null || ! command -v wget &>/dev/null || ! command -v curl &>/dev/null; then
    apt-get update -y && apt-get install -y zip unzip wget curl
else
    echo "    System tools already installed, skipping apt-get"
fi

# Install gdrive
if ! command -v gdrive &>/dev/null; then
    wget -q https://github.com/glotlabs/gdrive/releases/download/3.0.0/gdrive_linux-x64.tar.gz
    tar -xf gdrive_linux-x64.tar.gz
    chmod +x gdrive
    mv gdrive /usr/local/bin/
    rm gdrive_linux-x64.tar.gz
else
    echo "    gdrive already installed, skipping"
fi

# Install rclone
curl -s https://rclone.org/install.sh | bash || true

PIP="pip"
UVP="pip install --no-cache-dir"
PYBIN="python"

echo "=== [1/6] Removing conflicting packages ==="
pip uninstall -y torch torchvision torchaudio torch-scatter torch-sparse 2>/dev/null || true

echo "=== [2/6] Detecting CUDA version and installing PyTorch ==="
# Detect CUDA version from nvcc (major.minor)
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
else
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "12.4")
fi
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
echo "    Detected CUDA: ${CUDA_VER}"

# Choose PyTorch wheel: CUDA >= 12.8 → cu128 (supports sm_120 Blackwell)
#                       CUDA >= 12.4 → cu124
#                       fallback      → cu121
if [[ "$CUDA_MAJOR" -gt 12 ]] || [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 8 ]]; then
    TORCH_CUDA="cu128"
    TORCH_VER="2.7.0"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 4 ]]; then
    TORCH_CUDA="cu124"
    TORCH_VER="2.6.0"
else
    TORCH_CUDA="cu121"
    TORCH_VER="2.6.0"
fi
echo "    Installing PyTorch ${TORCH_VER} with ${TORCH_CUDA} into .venv"
$UVP torch==${TORCH_VER} torchvision \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

echo "=== [3/6] Installing PyG core ==="
$UVP torch-geometric

echo "=== [4/6] Installing PyG extensions (scatter / sparse) ==="
TORCH=$($PYBIN -c "import torch; print(torch.__version__.split('+')[0])")
CUDA=$($PYBIN -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
echo "    torch=${TORCH}  cuda=${CUDA}"
$UVP torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

echo "=== [5/6] Installing project dependencies ==="
$UVP \
  transformers loguru tqdm pyyaml \
  numpy pandas scikit-learn networkx \
  datasets sentencepiece

echo "=== [6/6] Installing project package ==="
$UVP -e .

echo ""
echo "=== Verification ==="
$PYBIN -c "
import torch
import torch_geometric
from torch_geometric.nn import GATv2Conv
from transformers import AutoModel
print(f'torch          : {torch.__version__}')
print(f'cuda available : {torch.cuda.is_available()}')
print(f'device         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
print(f'torch_geometric: {torch_geometric.__version__}')
print(f'GATv2Conv      : OK')
print(f'transformers   : OK')
print('All checks passed.')
"
