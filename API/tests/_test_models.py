"""Inference smoke test across all 3 registered architectures.

    python API/_test_models.py
Hits /health (shows device) then /inference for graph_based, hybrid_graph_lm,
sequential on the same functions, printing prediction + cls_embedding dim + latency.
Use after bringing the stack up (GPU: API_DEVICE=cuda) to mimic a real prod run.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
HERE = Path(__file__).resolve().parent
MODELS = ["graph_based", "hybrid_graph_lm", "sequential"]


def call(method: str, path: str, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    h = call("GET", "/health")
    print(f"== device={h['device']}  storage={h['storage']}  models={h['models']} ==")
    codes = json.loads((HERE / "_test_infer.json").read_text())["codes"]
    for mid in MODELS:
        t0 = time.time()
        try:
            res = call("POST", "/inference", {"model_id": mid, "codes": codes})
            dt = time.time() - t0
            print(f"\n[{mid}]  {dt:.1f}s")
            for r in res.get("results", []):
                print(f"   func[{r.get('index')}] ok={r.get('ok')} pred={r.get('prediction')} "
                      f"conf={r.get('confidence')} emb_dim={r.get('cls_embedding_dim')} "
                      f"err={r.get('error')}")
        except Exception as e:  # noqa: BLE001
            print(f"\n[{mid}]  ERROR  {type(e).__name__}: {e}")
    print("\n== done ==")


if __name__ == "__main__":
    main()
