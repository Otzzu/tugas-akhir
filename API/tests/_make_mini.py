"""Create a SMALL 26-class test dataset `megavul_mini` (subsample of full megavul) AND a
matching mini base model `graph_based_mini` (same n48 checkpoint, but pointed at
megavul_mini). Used by _test_relearn so EWC importance + ER replay run over the SMALL
task-A (megavul_mini), not the full 10716-graph megavul — fast + no OOM on a small box.

megavul_mini keeps megavul's exact 26-class vocab + label indexing → label-aligned with
the 26-class checkpoint; only the graph COUNT shrinks.

Run inside the api container:
    docker compose -f API/docker-compose.yml exec api python /app/API/tests/_make_mini.py [N]
N = graphs to keep (default 300). Idempotent: skips the heavy subsample if the .pt exists.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile

import torch

from gnn_vuln.config import Config
from gnn_vuln.data.dataset_lm import CodeBERTGraphDataset
from gnn_vuln.data.merge import _out_processed_path

from API.core.config import settings
from API.services import registry, storage

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ROOT = settings.ROOT / "data"
DEV = settings.DEVICE
N48_CONFIG = "API/configs/graph_based.yaml"
N48_CKPT = "checkpoints/graph_based.pt"


def _mini_cfg() -> Config:
    c = Config()
    c.data.mode = "multiclass"; c.data.storage = "inmemory"; c.data.max_nodes = 2500
    c.data.filter_top25_dangerous = False; c.data.max_per_class = 0; c.data.top_cwe = 0
    c.data.processed_dir = ROOT / "processed"
    c.model.pretrained_lm = "microsoft/unixcoder-base"
    c.model.func_lm = "microsoft/unixcoder-base"
    c.model.add_func_tokens = True; c.model.func_max_length = 1024
    return c


def main():
    mini = _mini_cfg()
    out_path = _out_processed_path(ROOT, "megavul_mini", mini)
    vdir = ROOT / "raw" / "megavul_mini"
    vocab_path = vdir / "cwe_vocab.json"

    if out_path.exists() and vocab_path.exists():
        print(f"megavul_mini already built ({out_path.name}); skipping subsample")
        vocab = json.loads(vocab_path.read_text())
        class_names = [""] * len(vocab)
        for k, i in vocab.items():
            class_names[i] = k
        uri = registry.get_dataset("megavul_mini").get("storage_uri") if _safe_get("megavul_mini") else None
    else:
        print("loading megavul (lazy)…")
        ds = CodeBERTGraphDataset(
            root=str(ROOT), source="megavul", mode="multiclass",
            pretrained_lm="microsoft/unixcoder-base", func_lm="microsoft/unixcoder-base",
            add_func_tokens=True, func_max_length=1024, max_nodes=2500,
            filter_top25_dangerous=True, max_per_class=1600, resample_seed=42,
            storage="lazy", embedder_device=DEV,
        )
        class_names = ds.class_names
        n = min(N, len(ds))
        graphs = [ds[i] for i in range(n)]
        print(f"megavul {len(ds)} -> mini {len(graphs)} graphs, {len(class_names)} classes")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"n_graphs": len(graphs), "class_names": class_names, "graphs": graphs}, out_path)
        vdir.mkdir(parents=True, exist_ok=True)
        vocab_path.write_text(json.dumps({n: i for i, n in enumerate(class_names)}, indent=2))

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(out_path, arcname="processed/" + out_path.name)
            tar.add(vocab_path, arcname="cwe_vocab.json")
        uri = storage.put_bytes(settings.S3_BUCKET_DATASETS, "megavul_mini.tar.gz", buf.getvalue())

    cfg = {
        "data": {"source": "megavul_mini", "mode": "multiclass", "max_nodes": 2500,
                 "storage": "inmemory", "filter_top25_dangerous": False, "max_per_class": 0,
                 "resample_seed": 42, "top_cwe": 0,
                 "raw_dir": str(ROOT / "raw"), "processed_dir": str(ROOT / "processed")},
        "model": {"pretrained_lm": "microsoft/unixcoder-base", "func_lm": "microsoft/unixcoder-base",
                  "add_func_tokens": True, "func_max_length": 1024, "num_classes": len(class_names)},
        "train": {"device": DEV},
    }
    data_config_id = registry.upsert_config("data", "megavul_mini", cfg)
    registry.register_dataset("megavul_mini", {
        "label": "megavul mini (test task-B, 26-class)", "source": "megavul_mini",
        "mode": "multiclass", "num_classes": len(class_names),
        "data_config_id": data_config_id, "storage_uri": uri, "storage": "inmemory",
        "max_nodes": 2500, "top_cwe": 0, "max_per_class": 0, "resample_seed": 42,
    })

    # mini base model: same n48 checkpoint, but dataset_id -> megavul_mini so EWC importance
    # + ER replay (which read base_ds.source) run over the SMALL set.
    registry.register_model("graph_based_mini", {
        "label": "graph_based mini base (tests)", "arch": "lmgat_codebert",
        "config": N48_CONFIG, "checkpoint": N48_CKPT, "dataset_id": "megavul_mini",
        "num_classes": len(class_names), "class_names": class_names,
    })
    print(f"registered dataset 'megavul_mini' + base 'graph_based_mini' ({len(class_names)} classes)")


def _safe_get(ds_id: str) -> bool:
    try:
        registry.get_dataset(ds_id); return True
    except KeyError:
        return False


if __name__ == "__main__":
    main()
