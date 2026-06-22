"""Relearn smoke test across ALL continual-learning methods.

    python API/_test_relearn.py [BASE_URL] [task_b_dataset_id]

Runs, in sequence, each method on the same base model + task-B dataset:
  finetune  — continue base weights, no protection
  EWC       — EWC-DR penalty (importance pass on the base dataset first)
  ER        — experience replay (buffer from the base dataset)
  EWC-ER    — both
retrain is already covered by _test_e2e.py.

Prereqs (continual methods need the BASE model's training data on local disk):
  - base model `graph_based` (n48) registered + checkpoint mounted.
  - base dataset `megavul` unzipped into ./data  (processed/<pt> + raw/megavul/cwe_vocab.json).
  - task-B dataset present too (defaults to megavul_26 — degenerate but exercises every
    code path; pass a real task-B id like the `relearn` dataset for a true forgetting run).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
TASK_B = sys.argv[2] if len(sys.argv) > 2 else "megavul_mini"  # small 26-class subset (fast)
BASE_MODEL = "graph_based_mini"  # mini base → EWC/ER replay over megavul_mini (no OOM)
METHODS = sys.argv[3].split(",") if len(sys.argv) > 3 else \
    ["retrain", "finetune", "EWC", "ER", "EWC-ER"]
EPOCHS = 1


def call(method: str, path: str, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_method(method: str):
    print(f"\n=== {method} (base={BASE_MODEL}, task_b={TASK_B}, epochs={EPOCHS}) ===")
    body = {"method": method, "dataset_ids": [TASK_B], "epochs": EPOCHS,
            "run_name": f"smoke_{method}"}
    if method != "retrain":
        body["base_model_id"] = BASE_MODEL
    j = call("POST", "/relearn", body)
    jid = j["job_id"]
    print("  job:", jid, j.get("status"))
    t0 = time.time()
    for _ in range(450):              # up to ~60 min (EWC importance + train on GPU)
        s = call("GET", "/relearn/" + jid)
        st = s.get("status")
        if st in ("done", "failed", "completed", "error"):
            print(f"  -> {st}  {int(time.time()-t0)}s  model={s.get('result_model_id')}  "
                  f"{s.get('message') or ''}")
            return st
        time.sleep(8)
    print("  -> TIMEOUT")
    return "timeout"


def main():
    h = call("GET", "/health")
    print(f"== device={h['device']}  models={h['models']} ==")
    results = {m: run_method(m) for m in METHODS}
    print("\n== summary ==")
    for m, st in results.items():
        print(f"  {m:8s} {st}")


if __name__ == "__main__":
    main()
