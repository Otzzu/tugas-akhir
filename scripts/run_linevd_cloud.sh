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
RUN_ID="linevd_megavul_ml1024${SEED:+_s$SEED}_$(date +%Y%m%d_%H%M%S)"
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
  # CUDA dgl (cu121) — Lightning moves the DGL graph to GPU, CPU dgl can't (DGLError
  # "Device API cuda is not enabled"). The cu121 wheel downloads fine (earlier 403 was
  # transient). dgl 2.4.0+cu121 cp312 pairs with torch 2.4.1+cu121.
  pip install -q dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
  # dgl install drags nvidia-nvjitlink-cu12 to a version that breaks torch's .so lookup — refix
  pip install -q --upgrade --force-reinstall torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
fi
# PL<2.0: LitGNN uses test_step->test_epoch_end(outputs) + Trainer(gpus=) — both REMOVED in PL 2.x.
# Force-reinstall in case the pod already has 2.x (model.all_funcs is built inside test_epoch_end).
pip install -q "pytorch-lightning<2.0" networkx fastparquet pydantic   # pydantic: DGL graphbolt dep
# torchmetrics 1.x eagerly imports torchvision; the pod's system torchvision (0.26+cu128,
# built for torch 2.11) ABI-mismatches our torch 2.4.1 -> `torchvision::nms does not exist`.
# Install the torch-2.4.1-matched build into the venv to shadow it. --no-deps keeps the pin.
pip install -q --no-deps torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# DGL's graphbolt imports torchdata.datapipes, REMOVED in torchdata>=0.10 -> pin older.
pip install -q --no-deps 'torchdata==0.9.0' || pip install -q --no-deps 'torchdata<0.10'
# rest of sastvd's actual runtime imports (per requirements.txt, minus dgl/torch already handled
# and torchtext/torchsummary/etc which are unused and would drag torch off the cu121 pin).
# transformers>=4.50 blocks torch.load of .bin checkpoints under torch<2.6 (CVE-2025-32434);
# codebert-base ships .bin and our torch is pinned 2.4.1 (DGL cap) -> pin transformers<4.50.
pip install -q gensim graphviz matplotlib networkx pandas scipy scikit-learn seaborn \
  torchmetrics tqdm "transformers<4.50" tsne-torch unidiff "ray[tune]" tensorboard   # tensorboard: sastvd/helpers/ml.py SummaryWriter
python -c "import torch,dgl; print('torch',torch.__version__,'dgl',dgl.__version__,'| cuda',torch.cuda.is_available())"

echo "=== [3/6] data + LineVD cache files (our split + flaw GT, no code edit) ==="
if [[ ! -d megavul_ml1024 ]]; then
  rclone copy "$REMOTE/data/baselines/$DATA_TAR" . --progress && tar -I "$(command -v pigz || echo gzip)" -xf "$DATA_TAR"
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
# LineVD hardcodes the "bigvul" dataset dir everywhere (dclass finished-glob, itempath,
# sast pkl, glove). Our graphs live under processed/megavul (dataset="megavul"). Bridge
# with symlinks so the hardcoded bigvul/ paths resolve — no rename, no rebuild, cache stays
# megavul-pathed. bigvul/ is a real dir (doc2vec writes there); only before/ + after/ link.
mkdir -p storage/processed/bigvul
ln -sfn ../megavul/before storage/processed/bigvul/before
ln -sfn ../megavul/after  storage/processed/bigvul/after
PYTHONPATH=. python sastvd/scripts/prepare.py

# LineVD item cache (built DGL graphs + per-func codebert embeddings + doc2vec). Building
# all ~14k graphs takes ~25min; restore from Drive on a fresh pod to go straight to train.
PREP_CACHE="linevd_megavul_prepcache.tar.gz"
PREP_MARK="storage/cache/bigvul_linevd_codebert_pdg+raw"
PREP_RESTORED=""
if [[ ! -d "$PREP_MARK" ]] && rclone ls "$REMOTE/data/baselines/$PREP_CACHE" >/dev/null 2>&1; then
  echo "  restoring LineVD prep cache from Drive (skips item build) ..."
  rclone copy "$REMOTE/data/baselines/$PREP_CACHE" /tmp/ --progress
  tar -I "${COMP[0]}" -xf "/tmp/$PREP_CACHE" && rm -f "/tmp/$PREP_CACHE"
  PREP_RESTORED=1
fi

echo "=== [5/6] train + localize ==="
OUT="$WORK/baseline_runs/$RUN_ID"; mkdir -p "$OUT"
# EVAL_ONLY=1 + WEIGHTS_TAR=<run>_weights.tar.gz -> restore the saved .ckpt and load it DIRECTLY
# (PL restores hparams from the ckpt), skipping the raytune train. Needs the prep cache (restored
# in step 4) so the test DataModule builds without re-parsing. Avoids the full retrain.
EVAL_ONLY="${EVAL_ONLY:-}"; WEIGHTS_TAR="${WEIGHTS_TAR:-}"; export LINEVD_CKPT=""
if [[ -n "$EVAL_ONLY" ]]; then
  [[ -n "$WEIGHTS_TAR" ]] || { echo "ERR: EVAL_ONLY=1 needs WEIGHTS_TAR=<...>_weights.tar.gz"; exit 1; }
  CKDIR="$WORK/linevd_eval_ckpt"; mkdir -p "$CKDIR"
  rclone copy "$REMOTE/checkpoints/baselines/$WEIGHTS_TAR" "$CKDIR/" --progress
  tar -I "${COMP[0]}" -xf "$CKDIR/$WEIGHTS_TAR" -C "$CKDIR" && rm -f "$CKDIR/$WEIGHTS_TAR"
  export LINEVD_CKPT="$(find "$CKDIR" -name '*.ckpt' | head -1)"
  [[ -n "$LINEVD_CKPT" ]] || { echo "ERR: no .ckpt found inside $WEIGHTS_TAR"; exit 1; }
  echo "  EVAL_ONLY: loaded $LINEVD_CKPT (skip raytune train)"
else
  export LINEVD_SEED="${SEED:-0}"
  PYTHONPATH=. python sastvd/scripts/train_best.py 2>&1 | tee "$OUT/train.log"
fi

# train_best.py ONLY trains — the test + statement-localization metrics come from rqtest
# (trainer.test on each best checkpoint -> get_relevant_metrics -> storage/outputs/rq_results_new/*.csv).
# rqtest.py is a while-True daemon; run ONE pass inline so the metrics actually get produced + uploaded.
echo "=== [5b/6] eval: rqtest native metrics + per-stmt dump for OUR localization metric ==="
export LINEVD_LOC_CSV="$OUT/linevd_loc_scores.csv"
PYTHONPATH=. python - <<'PYEOF' 2>&1 | tee "$OUT/eval.log" || true
import os
from glob import glob
import pandas as pd
import pytorch_lightning as pl
import sastvd as svd, sastvd.linevd as lvd
def mk_trainer():
    try:  # PL >=1.6 / 2.x API (the repo's gpus=1 was removed in PL 2.x)
        return pl.Trainer(accelerator="gpu", devices=1, logger=False, enable_checkpointing=False, default_root_dir="/tmp/")
    except TypeError:
        return pl.Trainer(gpus=1, default_root_dir="/tmp/")
ckpt = os.environ.get("LINEVD_CKPT")
if ckpt:
    # EVAL_ONLY: load the restored ckpt directly (hparams come from the ckpt) — no raytune needed
    print("EVAL_ONLY: loading", ckpt)
    model = lvd.LitGNN.load_from_checkpoint(ckpt, strict=False)
    gt = getattr(model.hparams, "gtype", "pdg+raw")
    emb = getattr(model.hparams, "embtype", "codebert")
    sp = getattr(model.hparams, "splits", "default")
    dm = lvd.BigVulDatasetLineVDDataModule(batch_size=1024, nsampling_hops=2, gtype=gt, splits=sp, feat=emb)
    # Manual test loop — bypass trainer.test/test_epoch_end (PL2.x removed it; their metric code also
    # breaks on numpy LongTensor). test_step returns [logits, labels, preds]; preds = per-func list.
    import torch as _th
    try: dm.setup("test")
    except Exception: pass
    dev = "cuda" if _th.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    all_funcs = []
    with _th.no_grad():
        for batch in dm.test_dataloader():
            try: batch = batch.to(dev)
            except Exception: pass
            out = model.test_step(batch, 0)
            all_funcs += out[2]
    model.all_funcs = all_funcs
    print("manual test done, funcs:", len(all_funcs))
else:
    from ray.tune import Analysis
    from sastvd.scripts.rqtest import main
    raytune_dirs = glob(str(svd.processed_dir() / "raytune_*_-1"))
    tune_dirs = [i for j in [glob(f"{rd}/*") for rd in raytune_dirs] for i in j]
    df = pd.concat([Analysis(d).dataframe() for d in tune_dirs])
    if "config/splits" not in df.columns: df["config/splits"] = "default"
    if "config/embtype" not in df.columns: df["config/embtype"] = "codebert"
    for c in df[["config/gtype", "config/splits", "config/embtype"]].drop_duplicates().to_dict("records"):
        main(c, df)
    print("EVAL DONE -> outputs/rq_results_new")
    best = df.loc[df["val_auroc"].astype(float).idxmax()]
    dm = lvd.BigVulDatasetLineVDDataModule(batch_size=1024, nsampling_hops=2,
            gtype=best["config/gtype"], splits=best["config/splits"], feat=best["config/embtype"])
    model = lvd.LitGNN.load_from_checkpoint(sorted(glob(best["logdir"] + "/checkpoint_*"))[-1] + "/checkpoint", strict=False)
    mk_trainer().test(model, dm)
rows = []
for fid, fn in enumerate(model.all_funcs):   # fn = [node_pred(softmax [N,2]), _VULN, pred_func, _LINE]
    sc, vu, _, ln = fn
    for j in range(len(ln)):
        rows.append((fid, int(ln[j]), float(sc[j][1]), int(vu[j])))
pd.DataFrame(rows, columns=["func_id", "line_number", "score", "is_flaw"]).to_csv(os.environ["LINEVD_LOC_CSV"], index=False)
print("LOC DUMP DONE ->", os.environ["LINEVD_LOC_CSV"], len(rows), "rows")
PYEOF
# recompute OUR localization metrics (IFA/Top-k/R@LOC/Effort) from the per-statement dump
if [[ -f "$LINEVD_LOC_CSV" ]]; then
  PYTHONPATH="$WORK/src" python "$WORK/scripts/compute_baseline_metrics.py" --name LineVD \
    --localization "$LINEVD_LOC_CSV" --out "$OUT/linevd_recomputed_metrics.json" 2>&1 | tee "$OUT/recomputed_metrics.log" || true
fi
# Cache the prep on Drive so future runs skip the d2v/glove rebuild. Upload when: prep exists AND
# (LINEVD_REFRESH_CACHE=1 to OVERWRITE a stale/incomplete cache, OR it was freshly built + not on Drive).
if [[ -d "$PREP_MARK" ]] && { [[ -n "${LINEVD_REFRESH_CACHE:-}" ]] || { [[ -z "$PREP_RESTORED" ]] && ! rclone ls "$REMOTE/data/baselines/$PREP_CACHE" >/dev/null 2>&1; }; }; then
  echo "  caching LineVD prep (codebert + d2v + glove) to Drive ..."
  tar -cf - storage/cache/codebert_method_level "$PREP_MARK" \
      storage/processed/bigvul/d2v_False storage/processed/bigvul/glove_False 2>/dev/null \
    | "${COMP[@]}" > "/tmp/$PREP_CACHE"
  rclone copyto "/tmp/$PREP_CACHE" "$REMOTE/data/baselines/$PREP_CACHE" --progress && rm -f "/tmp/$PREP_CACHE"
fi
# eval/localization metrics are in storage/outputs. do NOT copy storage/processed — that is the
# reusable graph cache (GBs, already uploaded to data/baselines as the prepcache) not a result.
cp -rf storage/outputs "$OUT/" 2>/dev/null || true
cd "$WORK"

echo "=== [6/6] upload weights + results ==="
tar -I "$(command -v pigz || echo gzip)" -cf "${RUN_ID}_results.tar.gz" -C "$OUT" .
rclone copy "${RUN_ID}_results.tar.gz" "$REMOTE/results/baselines/" --progress
WCKPT=$(find "$OUT" src/linevd/storage -name "*.ckpt" 2>/dev/null | head -1 || true)
if [[ -n "$WCKPT" ]]; then
  tar -I "$(command -v pigz || echo gzip)" -cf "${RUN_ID}_weights.tar.gz" -C "$(dirname "$WCKPT")" "$(basename "$WCKPT")"
  rclone copy "${RUN_ID}_weights.tar.gz" "$REMOTE/checkpoints/baselines/" --progress
fi
echo "DONE: $RUN_ID -> results/baselines/${RUN_ID}_results.tar.gz (+weights if any)"
