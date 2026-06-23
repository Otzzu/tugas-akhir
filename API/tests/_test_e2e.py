"""End-to-end smoke test — CHEAP path (seed only model 1 = graph_based + megavul_mini),
aiming for 100% functional coverage of the API surface.

Endpoints (all 18):
  GET  /health /models /datasets /configs /configs/{id}
  POST /inference /embed                         GET /inference/history /eval
  POST /datasets (ingest + merge)                GET /datasets/jobs /datasets/jobs/{id}
  POST /train                                    GET /train /train/{id}
  POST /relearn                                  GET /relearn /relearn/{id}

Functional paths: config-per-entity (NO kind), config-object bodies (inference top_k_lines,
datasets config, train/relearn config), data_config_id base + config override (layering),
dataset MERGE, /train from-scratch + config.model_type arch override, /relearn ALL FOUR methods
(finetune, ER, EWC, EWC-ER) with task-B label alignment, then inference on a relearned model.

    python API/tests/_test_e2e.py [BASE_URL]

Cheap seed first (host) — graph_based checkpoint + megavul_mini, skipping the big megavul_26:
    uv run --with gdown --with boto3 python API/scripts/seed_from_drive.py \
        --models graph_based --checkpoints-only

Stdlib only (urllib). Exits non-zero if any check fails. Deep DB+MinIO proof afterwards:
    docker compose -f API/docker-compose.yml exec api python /app/API/tests/_verify_persist.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
HERE = Path(__file__).resolve().parent
_fails: list[str] = []


def call(method: str, path: str, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode()[:500]}


def poll(path: str, every=8, limit=400):
    for _ in range(limit):
        j = call("GET", path)
        st = j.get("status")
        print(f"   …{path} -> {st}  {j.get('message') or ''}")
        if st in ("done", "failed", "completed", "error"):
            return j
        time.sleep(every)
    return j


def check(cond: bool, msg: str):
    print(f"   [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        _fails.append(msg)


def ingest(name: str, rows: list, *, data_config_id=None, config=None) -> str | None:
    body = {"name": name, "rows": rows}
    if data_config_id:
        body["data_config_id"] = data_config_id
    if config:
        body["config"] = config
    job = call("POST", "/datasets", body)
    jid = job.get("job_id"); print(f"   ingest '{name}' job:", jid, job.get("status"), job.get("_body") or "")
    if not jid:
        check(False, f"ingest '{name}' job not created ({job})"); return None
    done = poll(f"/datasets/jobs/{jid}")
    did = done.get("dataset_id")
    check(done.get("status") == "done" and bool(did), f"ingest '{name}' done -> {did}")
    return did


def main():
    print(f"== BASE {BASE} ==\n")

    # 1) health
    print("1) GET /health")
    check(call("GET", "/health").get("status") == "ok", "health ok")

    # 2) models — cheap seed: graph_based only
    print("2) GET /models")
    models = call("GET", "/models")
    check("graph_based" in models, f"graph_based seeded ({list(models)})")
    gb_cfg = models.get("graph_based", {}).get("config_id")
    check(bool(gb_cfg), f"graph_based has config_id ({gb_cfg})")

    # 3) configs — one per entity, NO kind
    print("3) GET /configs + /configs/{id} (no kind)")
    cfgs = call("GET", "/configs")
    check(bool(gb_cfg) and gb_cfg in cfgs, "graph_based config in /configs")
    check(all("kind" not in c for c in cfgs.values()), "no config has a `kind` field")
    if gb_cfg:
        one = call("GET", "/configs/" + urllib.parse.quote(gb_cfg, safe=""))
        check(bool(one.get("content")) and "kind" not in one, "config content retrievable, no kind")

    # 4) datasets — megavul_mini present with a data_config
    print("4) GET /datasets")
    dsets = call("GET", "/datasets")
    mini = dsets.get("megavul_mini", {})
    check(bool(mini), f"megavul_mini seeded ({list(dsets)})")
    mini_dcid = mini.get("data_config_id")
    check(bool(mini_dcid), f"megavul_mini has data_config_id ({mini_dcid})")

    # 5) inference — config.top_k_lines
    print("5) POST /inference (config.top_k_lines=3)")
    codes = json.loads((HERE / "_test_infer.json").read_text())
    res = call("POST", "/inference", {"model_id": "graph_based", **codes, "config": {"top_k_lines": 3}})
    ok = [r for r in res.get("results", []) if r.get("ok")]
    for r in res.get("results", []):
        print(f"   func[{r.get('index')}] ok={r.get('ok')} pred={r.get('prediction')} lines={len(r.get('suspicious_lines') or [])}")
    check(bool(ok), "inference produced a prediction")
    check(all(len(r.get("suspicious_lines") or []) <= 3 for r in ok), "config.top_k_lines=3 honored")

    # 6) embed
    print("6) POST /embed")
    e0 = next((r for r in call("POST", "/embed", {"model_id": "graph_based", **codes}).get("results", []) if r.get("ok")), {})
    vec = e0.get("cls_embedding") or []
    check(len(vec) > 0 and len(vec) == (e0.get("cls_embedding_dim") or -1), "embed returns pre-head vector")

    # 7) history + eval
    print("7) GET /inference/history + /eval")
    check(len(call("GET", "/inference/history?model_id=graph_based&limit=10")) >= 1, "inference history persisted")
    check("n_history" in call("GET", "/eval?model_id=graph_based"), "eval report returned")

    # 8) datasets ingest (rows) + 9) jobs list
    print("8) POST /datasets ingest A (rows)")
    rows = json.loads((HERE / "_test_ingest.json").read_text())["rows"]
    ds_a = ingest("e2e_a", rows)
    print("9) GET /datasets/jobs (list)")
    check(isinstance(call("GET", "/datasets/jobs"), list) and len(call("GET", "/datasets/jobs")) >= 1,
          "ingest jobs listed")
    if ds_a:
        rec = call("GET", "/datasets").get(ds_a, {})
        check(bool(rec.get("data_config_id")) and str(rec.get("storage_uri", "")).startswith("s3://"),
              "dataset A has data_config + object-storage bundle")

    # 10) ingest B reusing megavul_mini's data_config as base + a config override (layering)
    print("10) POST /datasets ingest B (data_config_id base + config override)")
    ds_b = ingest("e2e_b", rows, data_config_id=mini_dcid, config={"max_per_class": 5})

    # 11) MERGE A + B
    print("11) POST /datasets MERGE [A, B]")
    ds_merged = None
    if ds_a and ds_b:
        mj = call("POST", "/datasets", {"name": "e2e_merged", "dataset_ids": [ds_a, ds_b]})
        mjid = mj.get("job_id"); print("   merge job:", mjid, mj.get("status"), mj.get("_body") or "")
        if mjid:
            mdone = poll(f"/datasets/jobs/{mjid}")
            ds_merged = mdone.get("dataset_id")
            check(mdone.get("status") == "done" and bool(ds_merged), f"merge done -> {ds_merged}")
        else:
            check(False, f"merge job not created ({mj})")

    # 12) /train — from scratch on megavul_mini + config.model_type override
    print("12) POST /train (config_id=graph_based, megavul_mini, config.model_type)")
    tj = call("POST", "/train", {"config_id": gb_cfg, "dataset_ids": ["megavul_mini"],
                                 "run_name": "e2e_train", "config": {"epochs": 1, "model_type": "lmgat_codebert"}})
    tid = tj.get("job_id"); print("   train job:", tid, tj.get("status"), tj.get("_body") or "")
    trained = None
    if tid:
        trained = poll(f"/train/{tid}").get("result_model_id")
        check(bool(trained) and trained in call("GET", "/models"), f"train registered model ({trained})")
    else:
        check(False, f"train job not created ({tj})")
    print("   GET /train (list)")
    check(isinstance(call("GET", "/train"), list), "train jobs listed")

    # 13) /relearn — ALL FOUR CL methods, base = freshly trained model, task-B = dataset A (alignment)
    last_relearned = None
    if trained and ds_a:
        for method in ("finetune", "ER", "EWC", "EWC-ER"):
            print(f"13) POST /relearn ({method}, base=trained, task-B=A)")
            rj = call("POST", "/relearn", {"method": method, "base_model_id": trained,
                                           "dataset_ids": [ds_a], "run_name": f"e2e_{method}",
                                           "config": {"epochs": 1}})
            rid = rj.get("job_id"); print(f"   relearn[{method}] job:", rid, rj.get("status"), rj.get("_body") or "")
            if not rid:
                check(False, f"relearn[{method}] not created ({rj})"); continue
            rmid = poll(f"/relearn/{rid}").get("result_model_id")
            check(bool(rmid) and rmid in call("GET", "/models"), f"relearn[{method}] registered model ({rmid})")
            last_relearned = rmid or last_relearned
        print("   GET /relearn (list)")
        check(isinstance(call("GET", "/relearn"), list) and len(call("GET", "/relearn")) >= 4,
              "relearn jobs listed (>=4)")

    # 14) inference on a relearned model — full loop closes
    if last_relearned:
        print(f"14) POST /inference on relearned model ({last_relearned})")
        rres = call("POST", "/inference", {"model_id": last_relearned, **codes})
        check(any(r.get("ok") for r in rres.get("results", [])), "relearned model serves inference")

    print("\n== done ==")
    if _fails:
        print(f"FAILED ({len(_fails)}):")
        for f in _fails:
            print("  -", f)
        sys.exit(1)
    print("ALL FUNCTIONAL CHECKS PASSED.")
    print("Deep DB+MinIO proof:  docker compose -f API/docker-compose.yml exec api "
          "python /app/API/tests/_verify_persist.py")


if __name__ == "__main__":
    main()
