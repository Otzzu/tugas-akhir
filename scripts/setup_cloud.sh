#!/usr/bin/env bash
# Cloud GPU environment setup (RTX 4090, CUDA 12.4)
# Run once after each pod restart:
#   bash scripts/setup_cloud.sh

set -e

echo "=== [0/6] Installing system tools (zip, unzip, gdrive) ==="
apt-get update -y && apt-get install -y zip unzip wget
wget -q https://github.com/glotlabs/gdrive/releases/download/3.0.0/gdrive_linux-x64.tar.gz
tar -xf gdrive_linux-x64.tar.gz
chmod +x gdrive
mv gdrive /usr/local/bin/
rm gdrive_linux-x64.tar.gz

echo "=== [1/6] Removing conflicting packages ==="
pip uninstall -y torchaudio torch-scatter torch-sparse 2>/dev/null || true

echo "=== [2/6] Installing PyTorch 2.6 (CUDA 12.4) ==="
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "=== [3/6] Installing PyG core ==="
pip install --no-cache-dir torch-geometric

echo "=== [4/6] Installing PyG extensions (scatter / sparse) ==="
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
echo "    torch=${TORCH}  cuda=${CUDA}"
pip install --no-cache-dir torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

echo "=== [5/6] Installing project dependencies ==="
pip install --no-cache-dir \
  transformers loguru tqdm pyyaml \
  numpy pandas scikit-learn networkx \
  datasets sentencepiece

echo "=== [6/6] Installing project package ==="
pip install --no-cache-dir -e .

echo ""
echo "=== Verification ==="
python -c "
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
