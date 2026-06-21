"""End-to-end smoke test — exercises EVERY endpoint AND asserts the data actually
persisted (DB rows + object-storage pointers), not just HTTP 200s.

    python API/tests/_test_e2e.py [BASE_URL]

Covers all 13 routes: /health /models /inference /embed /inference/history /eval
/datasets(POST,GET) /datasets/jobs(GET,{id}) /configs(GET,{id}) /relearn(POST,GET,{id}).
Stdlib only (urllib). Exits non-zero if any check fails.

For the deep DB+MinIO proof (head_object on every stored pointer) run, after this:
    docker compose -f API/docker-compose.yml exec api python /app/API/tests/_verify_persist.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
HERE = Path(__file__).resolve().parent
_fails: list[str] = []


def call(method: str, path: str, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


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

    # 1) GET /health
    print("1) GET /health")
    h = call("GET", "/health")
    print("  ", h)
    check(h.get("status") == "ok", "health ok")

    # 2) GET /models
    print("2) GET /models")
    models = call("GET", "/models")
    check(all(m in models for m in ("graph_based", "hybrid_graph_lm", "sequential")),
          f"3 seed models registered ({list(models)})")

    # 3) POST /inference
    print("3) POST /inference (graph_based)")
    codes = json.loads((HERE / "_test_infer.json").read_text())
    res = call("POST", "/inference", {"model_id": "graph_based", **codes})
    for r in res.get("results", []):
        print(f"   func[{r.get('index')}] ok={r.get('ok')} pred={r.get('prediction')} "
              f"conf={r.get('confidence')} emb_dim={r.get('cls_embedding_dim')} err={r.get('error')}")
    check(any(r.get("ok") for r in res.get("results", [])), "inference produced a prediction")

    # 4) POST /embed
    print("4) POST /embed (graph_based)")
    emb = call("POST", "/embed", {"model_id": "graph_based", **codes})
    e0 = next((r for r in emb.get("results", []) if r.get("ok")), {})
    vec = e0.get("cls_embedding") or []
    print(f"   emb_dim={e0.get('cls_embedding_dim')} vector_len={len(vec)}")
    check(len(vec) > 0 and len(vec) == (e0.get("cls_embedding_dim") or -1),
          "embed returns the pre-head vector")

    # 5) GET /inference/history  (persistence: predictions+embedding in DB)
    print("5) GET /inference/history")
    hist = call("GET", "/inference/history?model_id=graph_based&limit=10")
    check(len(hist) >= 1, f"inference_results persisted in DB ({len(hist)} row[s])")
    check(any(x.get("cls_embedding_dim") for x in hist), "stored rows carry cls_embedding_dim")

    # 6) GET /eval
    print("6) GET /eval")
    rep = call("GET", "/eval?model_id=graph_based")
    print("  ", rep)
    check("n_history" in rep, "eval report returned")

    # 7) POST /datasets  (+ 8 poll job, 9 jobs list, 10 datasets list, 11/12 configs)
    print("7) POST /datasets (ingest)")
    ing = json.loads((HERE / "_test_ingest.json").read_text())
    job = call("POST", "/datasets", ing)
    jid = job["job_id"]; print("   job:", jid, job["status"])
    done = poll(f"/datasets/jobs/{jid}")
    ds_id = done.get("dataset_id")
    check(done.get("status") == "done" and bool(ds_id), f"ingest finished, dataset_id={ds_id}")

    if ds_id:
        print("8/9) GET /datasets/jobs (list)")
        jobs = call("GET", "/datasets/jobs")
        check(any(j.get("job_id") == jid for j in jobs), "ingest job listed")

        print("10) GET /datasets — bundle pointer to object storage")
        dsets = call("GET", "/datasets")
        rec = dsets.get(ds_id, {})
        uri = rec.get("storage_uri", "")
        check(bool(rec), f"dataset {ds_id} registered in DB")
        check(uri.startswith("s3://"), f"dataset bundle in object storage ({uri})")

        print("11/12) GET /configs?kind=data + /configs/{id}")
        dcid = rec.get("data_config_id")
        cfgs = call("GET", "/configs?kind=data")
        check(bool(dcid) and dcid in cfgs, f"immutable data-config registered ({dcid})")
        if dcid:
            one = call("GET", "/configs/" + urllib.parse.quote(dcid, safe=""))
            check(bool(one.get("content")), "data-config content retrievable by id")

        # 13) POST /relearn + poll + list
        print("13) POST /relearn (retrain) + GET /relearn/{id} + GET /relearn")
        rj = call("POST", "/relearn", {"method": "retrain", "dataset_ids": [ds_id],
                                       "epochs": 1, "run_name": "e2e_smoke"})
        rid = rj["job_id"]; print("   relearn job:", rid, rj.get("status"))
        rdone = poll(f"/relearn/{rid}")
        mid = rdone.get("result_model_id")
        check(rdone.get("status") in ("done", "completed"), "relearn job finished")
        check(bool(mid) and mid in call("GET", "/models"),
              f"relearned model registered in DB ({mid})")
        rlist = call("GET", "/relearn")
        check(any(j.get("job_id") == rid for j in rlist), "relearn job listed")

    print("\n== done ==")
    if _fails:
        print(f"FAILED ({len(_fails)}):")
        for f in _fails:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL ENDPOINT + PERSISTENCE CHECKS PASSED (DB).")
    print("Deep DB+MinIO proof:  docker compose -f API/docker-compose.yml exec api "
          "python /app/API/tests/_verify_persist.py")


if __name__ == "__main__":
    main()
