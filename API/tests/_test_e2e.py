"""End-to-end smoke test — CHEAP path (seed only model 1 = graph_based + megavul_mini).

Exercises the post-refactor API and asserts persistence (DB rows + object-storage pointers),
not just HTTP 200s:
  - config-per-entity, NO kind taxonomy (GET /configs has no `kind` field)
  - config-object request bodies (inference/datasets/train/relearn separate config from job spec)
  - POST /train  — from-scratch + `config.model_type` architecture override
  - POST /relearn — continue a base model, with task-B label alignment

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


def poll(path: str, every=8, limit=300):
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


def main():
    print(f"== BASE {BASE} ==\n")

    # 1) health
    print("1) GET /health")
    h = call("GET", "/health")
    check(h.get("status") == "ok", f"health ok ({h.get('status')})")

    # 2) models — cheap seed: only graph_based required
    print("2) GET /models")
    models = call("GET", "/models")
    check("graph_based" in models, f"graph_based seeded ({list(models)})")
    gb_cfg = models.get("graph_based", {}).get("config_id")
    check(bool(gb_cfg), f"graph_based has a config_id ({gb_cfg})")

    # 3) configs — ONE config per entity, NO kind taxonomy
    print("3) GET /configs (no kind)")
    cfgs = call("GET", "/configs")
    check(bool(gb_cfg) and gb_cfg in cfgs, "graph_based config_id is in /configs")
    check(all("kind" not in c for c in cfgs.values()), "no config carries a `kind` field")
    if gb_cfg:
        one = call("GET", "/configs/" + urllib.parse.quote(gb_cfg, safe=""))
        check(bool(one.get("content")) and "kind" not in one, "config content retrievable, no kind")

    # 4) datasets — megavul_mini present (cheap test data) + has a data_config
    print("4) GET /datasets")
    dsets = call("GET", "/datasets")
    check("megavul_mini" in dsets, f"megavul_mini seeded ({list(dsets)})")
    check(bool(dsets.get("megavul_mini", {}).get("data_config_id")),
          f"megavul_mini has a data_config_id ({dsets.get('megavul_mini', {}).get('data_config_id')})")

    # 5) inference — config object (top_k_lines)
    print("5) POST /inference (graph_based, config.top_k_lines=3)")
    codes = json.loads((HERE / "_test_infer.json").read_text())
    res = call("POST", "/inference", {"model_id": "graph_based", **codes, "config": {"top_k_lines": 3}})
    for r in res.get("results", []):
        print(f"   func[{r.get('index')}] ok={r.get('ok')} pred={r.get('prediction')} "
              f"conf={r.get('confidence')} lines={len(r.get('suspicious_lines') or [])}")
    ok_results = [r for r in res.get("results", []) if r.get("ok")]
    check(bool(ok_results), "inference produced a prediction")
    check(all(len(r.get("suspicious_lines") or []) <= 3 for r in ok_results), "config.top_k_lines=3 honored")

    # 6) embed
    print("6) POST /embed")
    emb = call("POST", "/embed", {"model_id": "graph_based", **codes})
    e0 = next((r for r in emb.get("results", []) if r.get("ok")), {})
    vec = e0.get("cls_embedding") or []
    check(len(vec) > 0 and len(vec) == (e0.get("cls_embedding_dim") or -1), "embed returns the pre-head vector")

    # 7) history + eval (persistence)
    print("7) GET /inference/history + /eval")
    hist = call("GET", "/inference/history?model_id=graph_based&limit=10")
    check(len(hist) >= 1, f"inference_results persisted ({len(hist)} row[s])")
    check("n_history" in call("GET", "/eval?model_id=graph_based"), "eval report returned")

    # 8) datasets ingest — config object defaults + data_config registered
    print("8) POST /datasets (ingest)")
    ing = json.loads((HERE / "_test_ingest.json").read_text())
    job = call("POST", "/datasets", ing)
    jid = job.get("job_id"); print("   job:", jid, job.get("status"), job.get("_body") or "")
    done = poll(f"/datasets/jobs/{jid}")
    ds_id = done.get("dataset_id")
    check(done.get("status") == "done" and bool(ds_id), f"ingest finished, dataset_id={ds_id}")
    if ds_id:
        rec = call("GET", "/datasets").get(ds_id, {})
        dcid = rec.get("data_config_id")
        check(bool(dcid) and dcid in call("GET", "/configs"), f"ingest data-config registered ({dcid})")
        check(str(rec.get("storage_uri", "")).startswith("s3://"), "dataset bundle in object storage")

    # 9) /train — from scratch on megavul_mini, config object + model_type override
    print("9) POST /train (config_id=graph_based, dataset=megavul_mini, config.model_type=lmgat_codebert)")
    tj = call("POST", "/train", {
        "config_id": gb_cfg, "dataset_ids": ["megavul_mini"], "run_name": "e2e_train",
        "config": {"epochs": 1, "model_type": "lmgat_codebert"}})
    tid = tj.get("job_id"); print("   train job:", tid, tj.get("status"), tj.get("_body") or "")
    trained = None
    if tid:
        tdone = poll(f"/train/{tid}")
        trained = tdone.get("result_model_id")
        check(tdone.get("status") in ("done", "completed"), "train job finished")
        check(bool(trained) and trained in call("GET", "/models"), f"trained model registered ({trained})")
    else:
        check(False, f"train job not created ({tj})")

    # 10) /relearn — continue the freshly-trained model on megavul_mini (config object + alignment)
    if trained:
        print("10) POST /relearn (finetune, base=trained model, dataset=megavul_mini)")
        rj = call("POST", "/relearn", {
            "method": "finetune", "base_model_id": trained, "dataset_ids": ["megavul_mini"],
            "run_name": "e2e_relearn", "config": {"epochs": 1}})
        rid = rj.get("job_id"); print("   relearn job:", rid, rj.get("status"), rj.get("_body") or "")
        if rid:
            rdone = poll(f"/relearn/{rid}")
            mid = rdone.get("result_model_id")
            check(rdone.get("status") in ("done", "completed"), "relearn job finished")
            check(bool(mid) and mid in call("GET", "/models"), f"relearned model registered ({mid})")
        else:
            check(False, f"relearn job not created ({rj})")

    print("\n== done ==")
    if _fails:
        print(f"FAILED ({len(_fails)}):")
        for f in _fails:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED.")
    print("Deep DB+MinIO proof:  docker compose -f API/docker-compose.yml exec api "
          "python /app/API/tests/_verify_persist.py")


if __name__ == "__main__":
    main()
