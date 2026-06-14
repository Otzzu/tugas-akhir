#!/usr/bin/env bash
# run_linevd_cloud.sh — train + localize LineVD on OUR MegaVul split, upload to Drive.
# PILOT: LineVD = DGL + PyTorch-Lightning + Joern (2022 repo). Expect 1-2 pod iterations
# on the DGL<->CUDA version match and Joern setup. The data path is locked (our split +
# our flaw GT via linevd_prepare_megavul.py); what may need tuning is the env, not the data.
#
# Usage (pod, from project root):
#   bash scripts/run_linevd_cloud.sh
set -euo pipefail

REMOTE="gdrive-mesach:tugas-akhir"
DATA_TAR="megavul_ml1024_baselines_20260613.tar.gz"
RUN_ID="linevd_megavul_ml1024_$(date +%Y%m%d_%H%M%S)"
WORK="$PWD"

echo "=== [1/6] LineVD present (vendored in-repo; clone only if missing) ==="
[[ -d src/linevd ]] || git clone --depth 1 https://github.com/davidhin/linevd.git src/linevd

echo "=== [2/6] isolated venv (torch 2.4.1+cu121 + dgl 2.4.0; pod main env torch 2.9 has no DGL build) ==="
VENV="/workspace/linevd_env"
if [[ ! -d "$VENV" ]]; then
  /venv/main/bin/python -m venv "$VENV" --system-site-packages   # 3.11, inherits pod pkgs (transformers/pandas/...)
fi
source "$VENV/bin/activate"
if ! python -c "import torch,dgl" 2>/dev/null; then
  pip install -q torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
  # data.dgl.ai CUDA wheels (cu121, any dgl version) 403 — CPU dgl 2.4.0's .so covers
  # torch 2.4.0 AND 2.4.1, so CPU dgl + cu121 torch works fine (only DGL ops run on CPU).
  pip install -q dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
  # dgl install drags nvidia-nvjitlink-cu12 to a version that breaks torch's .so lookup — refix
  pip install -q --upgrade --force-reinstall torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
fi
pip install -q pytorch-lightning networkx fastparquet pydantic   # pydantic: DGL graphbolt dep
# torchmetrics 1.x eagerly imports torchvision; the pod's system torchvision (0.26+cu128,
# built for torch 2.11) ABI-mismatches our torch 2.4.1 -> `torchvision::nms does not exist`.
# Install the torch-2.4.1-matched build into the venv to shadow it. --no-deps keeps the pin.
pip install -q --no-deps torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# DGL's graphbolt imports torchdata.datapipes, REMOVED in torchdata>=0.10 -> pin older.
pip install -q --no-deps 'torchdata==0.9.0' || pip install -q --no-deps 'torchdata<0.10'
# rest of sastvd's actual runtime imports (per requirements.txt, minus dgl/torch already handled
# and torchtext/torchsummary/etc which are unused and would drag torch off the cu121 pin).
pip install -q gensim graphviz matplotlib networkx pandas scipy scikit-learn seaborn \
  torchmetrics tqdm transformers tsne-torch unidiff "ray[tune]" tensorboard   # tensorboard: sastvd/helpers/ml.py SummaryWriter
python -c "import torch,dgl; print('torch',torch.__version__,'dgl',dgl.__version__,'| cuda',torch.cuda.is_available())"

echo "=== [3/6] data + LineVD cache files (our split + flaw GT, no code edit) ==="
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar -xzf "$DATA_TAR"
fi
cd src/linevd
PYTHONPATH=. python "$WORK/scripts/linevd_prepare_megavul.py" --in-dir "$WORK/megavul_ml1024/linevd"

echo "=== [4/6] Joern + build graphs + codebert embeddings ==="
# Joern graphs (.edges.json/.nodes.json per func) take ~3h to build. Cache the whole
# processed/megavul tree on Drive so a fresh pod restores it and skips getgraphs.
GRAPH_CACHE="linevd_megavul_joerngraphs.tar.gz"
command -v pigz >/dev/null || apt-get install -y -q pigz || true   # parallel gz
# NB: `cmd && COMP=(...)` as a bare statement trips set -e when cmd fails — use if.
if command -v pigz >/dev/null; then COMP=(pigz -p "$(nproc)"); else COMP=(gzip); fi
HAVE=$( (ls storage/processed/megavul/before/*.edges.json 2>/dev/null || true) | wc -l )
if [[ "$HAVE" -lt 1000 ]] && rclone ls "$REMOTE/data/baselines/$GRAPH_CACHE" >/dev/null 2>&1; then
  echo "  restoring cached joern graphs from Drive (skips getgraphs) ..."
  rclone copy "$REMOTE/data/baselines/$GRAPH_CACHE" /tmp/ --progress
  tar -I "${COMP[0]}" -xf "/tmp/$GRAPH_CACHE" && rm -f "/tmp/$GRAPH_CACHE"
fi
BEFORE_CNT=$( (ls storage/processed/megavul/before/*.edges.json 2>/dev/null || true) | wc -l )

# LineVD wraps every subprocess in `singularity exec main.sif` unless SINGULARITY=true.
# We run joern/flawfinder directly on the pod, so bypass the container wrapper.
export SINGULARITY=true
# Joern v1.1.260 (2022) bundles a Scala/ammonite that cannot parse Java 21 classfiles
# (ConstantPool errorBadIndex crash). Pin Java 17. Outside the download block so it
# always exports JAVA_HOME, even when joern-cli was unzipped on a prior run.
JAVA17="/usr/lib/jvm/java-17-openjdk-amd64"
[[ -d "$JAVA17" ]] || (apt-get update -q && apt-get install -y -q openjdk-17-jre-headless)
export JAVA_HOME="$JAVA17"
export PATH="$JAVA_HOME/bin:$PATH"
# svd.external_dir() = storage/external/joern-cli — old pinned release the LineVD code expects.
if [[ ! -d storage/external/joern-cli ]]; then
  wget -q https://github.com/joernio/joern/releases/download/v1.1.260/joern-cli.zip -O /tmp/joern-cli.zip
  mkdir -p storage/external
  unzip -q /tmp/joern-cli.zip -d storage/external
  rm -f /tmp/joern-cli.zip
fi
# Ammonite exports the JDK runtime to ~/.ammonite/rt-<ver>.jar on first boot. getgraphs
# runs 8 parallel joern workers that all race to create it (FileAlreadyExistsException).
# Pre-warm single-threaded with the REAL script on a throwaway C file (exact compile path
# the workers hit) so the jar exists before the parallel loop; retry until it lands.
rm -f /root/.ammonite/rt-*.jar
printf 'int warm(int a){return a+1;}\n' > /tmp/joern_warm.c
for _ in 1 2 3; do
  [[ -e /root/.ammonite/rt-*.jar ]] 2>/dev/null && break
  storage/external/joern-cli/joern --script storage/external/get_func_graph.scala \
    --params='filename=/tmp/joern_warm.c' || true
  ls /root/.ammonite/rt-*.jar >/dev/null 2>&1 && break
done
ls -la /root/.ammonite/rt-*.jar 2>&1 || echo "WARN: rt jar not pre-warmed; parallel joern may race"
# getgraphs is a 100-way job-array script (NUM_JOBS=100); loop all shards to cover our full split.
# Idempotent: skips any func whose .edges.json already exists (so a restored cache is a no-op).
for i in $(seq 1 100); do PYTHONPATH=. python sastvd/scripts/getgraphs.py "$i"; done

AFTER_CNT=$( (ls storage/processed/megavul/before/*.edges.json 2>/dev/null || true) | wc -l )
# Refresh the Drive cache only if we built new graphs (or no cache exists yet).
if [[ "$AFTER_CNT" -gt "$BEFORE_CNT" ]] || ! rclone ls "$REMOTE/data/baselines/$GRAPH_CACHE" >/dev/null 2>&1; then
  echo "  caching joern graphs to Drive ($AFTER_CNT funcs) ..."
  tar -cf - storage/processed/megavul | "${COMP[@]}" > "/tmp/$GRAPH_CACHE"
  rclone copy "/tmp/$GRAPH_CACHE" "$REMOTE/data/baselines/" --progress && rm -f "/tmp/$GRAPH_CACHE"
fi
PYTHONPATH=. python sastvd/scripts/prepare.py

echo "=== [5/6] train + localize ==="
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
PYTHONPATH=. python sastvd/scripts/train_best.py 2>&1 | tee "$OUT/train.log"
# eval/localization metrics are emitted by linevd's eval; copy storage outputs
cp -rf storage/processed "$OUT/" 2>/dev/null || true
cp -rf storage/outputs "$OUT/" 2>/dev/null || true
cd "$WORK"

echo "=== [6/6] upload weights + results ==="
tar -czf "${RUN_ID}_results.tar.gz" -C "$OUT" .
rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress
WCKPT=$(find "$OUT" src/linevd/storage -name "*.ckpt" 2>/dev/null | head -1 || true)
if [[ -n "$WCKPT" ]]; then
  tar -czf "${RUN_ID}_weights.tar.gz" -C "$(dirname "$WCKPT")" "$(basename "$WCKPT")"
  rclone copy "${RUN_ID}_weights.tar.gz" "$REMOTE/checkpoints/baselines/" --progress
fi
echo "DONE: $RUN_ID -> results/baselines/${RUN_ID}_results.tar.gz (+weights if any)"
