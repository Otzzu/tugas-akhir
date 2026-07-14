#!/usr/bin/env bash
# make_mini_bundle.sh
# ~~~~~~~~~~~~~~~~~~~
# megavul_mini (dataset kecil untuk demo + E2E) dari dataset NINE resmi.
#
# Mini cuma SUBSAMPLE graph yang sudah jadi — tidak ada Joern, tidak ada embedding ulang,
# tidak butuh GPU. Featurization-nya otomatis mengikuti sumbernya (nine), jadi ia sepadan
# dengan model resmi dan dengan dataset hasil ingest API.
#
# Keluaran: api_datasets/megavul_mini/megavul_mini.tar.gz di Drive (format API).
#
# Run (pod, rclone terpasang):
#   bash scripts/make_mini_bundle.sh [N_PER_CLASS]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

N="${1:-15}"                      # graph per kelas (26 kelas -> ~300 graph)
DRIVE="gdrive-mesach:tugas-akhir"
SRC="lm_dataset_megavul_multiclass_unixcoder-base-nine_ft_ml1024_f40f2e964_s1600r42"
PROC="data/processed"
COMP="$(command -v pigz || echo gzip)"

command -v pigz >/dev/null || (apt-get update -q && apt-get install -y -q pigz) || true
mkdir -p "$PROC"

echo "=== [1/4] unduh dataset nine (kalau belum ada) ==="
if [[ ! -f "$PROC/${SRC}_meta.pt" ]]; then
  TAR=$(rclone lsf "$DRIVE/data/processed/megavul" | grep "^${SRC}_lazy_.*\.tar\.gz$" | sort | tail -1)
  [[ -n "$TAR" ]] || { echo "ERR: tar nine tidak ketemu di Drive"; exit 1; }
  rclone copy "$DRIVE/data/processed/megavul/$TAR" "$PROC/" --progress
  tar -I "$COMP" -xf "$PROC/$TAR" -C "$PROC"
  rm -f "$PROC/$TAR"
fi
echo "  sumber: ${SRC}_meta.pt"

echo "=== [2/4] subsample $N graph per kelas ==="
python - "$SRC" "$N" <<'PY'
import json, sys, torch
from collections import defaultdict
from pathlib import Path
from gnn_vuln.core import open_dataset

src, n_per = sys.argv[1], int(sys.argv[2])
proc = Path("data/processed")
ds = open_dataset(proc / f"{src}_meta.pt")

per = defaultdict(list)
for i in range(len(ds)):
    y = int(ds[i].y)
    if len(per[y]) < n_per:
        per[y].append(i)
graphs = [ds[i] for idxs in per.values() for i in idxs]
names = list(ds.class_names)
print(f"  {len(ds)} -> {len(graphs)} graph, {len(names)} kelas, {len(per)} kelas terisi")

out = proc / "lm_dataset_megavul_mini_multiclass_unixcoder-base-nine_ft_ml1024.pt"
torch.save({"n_graphs": len(graphs), "class_names": names,
            "fingerprint": {"schema": 1, "source": "megavul_mini", "mode": "multiclass",
                            "storage": "inmemory", "max_nodes": 2500,
                            "pretrained_lm": "microsoft/unixcoder-base-nine",
                            "func_lm": "microsoft/unixcoder-base-nine",
                            "add_func_tokens": True, "func_max_length": 1024,
                            "node_feat_dim": int(graphs[0].x.shape[1]),
                            "edge_dim": int(graphs[0].edge_attr.shape[1])},
            "graphs": graphs}, out)
Path("data/raw/megavul_mini").mkdir(parents=True, exist_ok=True)
Path("data/raw/megavul_mini/cwe_vocab.json").write_text(
    json.dumps({c: i for i, c in enumerate(names)}, indent=2), encoding="utf-8")
print(f"  -> {out.name} ({out.stat().st_size/1e6:.0f} MB)")
PY

echo "=== [3/4] bundle format API ==="
STAGE="data/_mini"; rm -rf "$STAGE"; mkdir -p "$STAGE/processed"
cp "$PROC/lm_dataset_megavul_mini_multiclass_unixcoder-base-nine_ft_ml1024.pt" "$STAGE/processed/"
cp data/raw/megavul_mini/cwe_vocab.json "$STAGE/"
# anggota disebut eksplisit — `tar -c .` menambahkan awalan "./" dan materialize_dataset
# menganggapnya bukan format API, lalu mengubur .pt di processed/processed/
tar -C "$STAGE" -cf - cwe_vocab.json processed | $COMP > "$PROC/megavul_mini.tar.gz"
rm -rf "$STAGE"
echo "  -> megavul_mini.tar.gz ($(du -h "$PROC/megavul_mini.tar.gz" | cut -f1))"
tar -tzf "$PROC/megavul_mini.tar.gz" | head -3 | sed 's/^/     /'

echo "=== [4/4] unggah ==="
rclone copy "$PROC/megavul_mini.tar.gz" "$DRIVE/api_datasets/megavul_mini/" --progress
rm -f "$PROC/megavul_mini.tar.gz"
echo "SELESAI -> $DRIVE/api_datasets/megavul_mini/megavul_mini.tar.gz"
