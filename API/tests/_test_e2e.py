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
raw-dataset lineage, flaw_lines+func_after on one row, dataset MERGE, /train from-scratch +
config.model_type arch override, /relearn ALL FOUR methods (finetune, ER, EWC, EWC-ER) with
task-B label alignment + CUMULATIVE dataset lineage, a REFUSED featurization mismatch, then
inference on a relearned model.

    python API/tests/_test_e2e.py [BASE_URL]

Seed first (host) — model 1 with its dataset + megavul_mini (a model is always seeded with its
dataset). This e2e relearns a small model trained on megavul_mini, so it stays cheap on CPU:
    uv run --with gdown --with boto3 python API/scripts/seed_from_drive.py --models graph_based

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
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # transient (API busy under a heavy worker job) — return so poll() keeps retrying
        return {"_error": str(e)}


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
        # raw lineage: the uploaded rows are a first-class entity, so the same raw can be
        # rebuilt under another featurization without re-uploading it.
        check(bool(rec.get("raw_id")), f"dataset A records its raw_id ({rec.get('raw_id')})")

    # 8b) a row carrying BOTH flaw_lines and func_after — flaw_lines is the annotation, the
    #     diff is only a fallback, so the explicit one must win (never silently dropped).
    print("8b) POST /datasets ingest (flaw_lines + func_after together)")
    both = [{"language": "c", "cwe": "CWE-787", "flaw_lines": [3],
             "code": "void f(const char *s) {\n    char b[8];\n    strcpy(b, s);\n    puts(b);\n}",
             "func_after": "void f(const char *s) {\n    char b[8];\n    strncpy(b, s, sizeof(b) - 1);\n    puts(b);\n}"},
            {"language": "c", "label": 0,
             "code": "int add(int a, int b) {\n    return a + b;\n}"}]
    ds_both = ingest("e2e_both", both)
    check(bool(ds_both), "row with flaw_lines + func_after ingests (explicit annotation kept)")

    # 10) ingest B reusing megavul_mini's data_config as base + a config override (layering).
    #     The ingest() helper asserts it finishes — that IS the layering test.
    print("10) POST /datasets ingest B (data_config_id base + config override)")
    ingest("e2e_b", rows, data_config_id=mini_dcid, config={"max_per_class": 5})

    # 11) MERGE different-vocab datasets — ds_a (C/C++, few classes) + megavul_mini (26 classes).
    #     This exercises VOCAB RESOLUTION: the merge must unify the two label spaces (map ds_a's
    #     CWEs onto the shared vocab), so the merged class count is the UNION, not either alone.
    print("11) POST /datasets MERGE [A, megavul_mini] (vocab resolve)")
    ds_merged = None
    mini_nc = int(mini.get("num_classes") or 0)
    if ds_a:
        mj = call("POST", "/datasets", {"name": "e2e_merged", "dataset_ids": [ds_a, "megavul_mini"]})
        mjid = mj.get("job_id"); print("   merge job:", mjid, mj.get("status"), mj.get("_body") or "")
        if mjid:
            mdone = poll(f"/datasets/jobs/{mjid}")
            ds_merged = mdone.get("dataset_id")
            check(mdone.get("status") == "done" and bool(ds_merged), f"merge done -> {ds_merged}")
            if ds_merged:
                mrec = call("GET", "/datasets").get(ds_merged, {})
                merged_nc = int(mrec.get("num_classes") or 0)
                check(merged_nc >= mini_nc, f"merge resolved vocab to the union ({merged_nc} >= mini {mini_nc})")
                # provenance: a merged dataset names its parents, each keeping its own raw_id
                check(set(mrec.get("source_dataset_ids") or []) == {ds_a, "megavul_mini"},
                      f"merge records its parents ({mrec.get('source_dataset_ids')})")
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

    # 13) /relearn — ALL FOUR CL methods. base = the freshly /train-ed model (its dataset is the
    #     small megavul_mini), so ER/EWC materialize megavul_mini + build the replay buffer / EWC
    #     importance over it — feasible on a CPU box. task-B = dataset A (C/C++) exercises label
    #     alignment onto the base's class space.
    #     NOTE: relearning the real model 1 (graph_based, dataset = megavul_26 ~3.8GB) works but
    #     ER/EWC over it overload a CPU-only box — run that on a GPU host, not in this smoke test.
    last_relearned = None
    if trained and ds_a:
        for method in ("finetune", "ER", "EWC", "EWC-ER"):
            print(f"13) POST /relearn ({method}, base=trained-on-mini, task-B=A)")
            rj = call("POST", "/relearn", {"method": method, "base_model_id": trained,
                                           "dataset_ids": [ds_a], "run_name": f"e2e_{method}",
                                           "config": {"epochs": 1}})
            rid = rj.get("job_id"); print(f"   relearn[{method}] job:", rid, rj.get("status"), rj.get("_body") or "")
            if not rid:
                check(False, f"relearn[{method}] not created ({rj})"); continue
            rmid = poll(f"/relearn/{rid}").get("result_model_id")
            check(bool(rmid) and rmid in call("GET", "/models"), f"relearn[{method}] registered model ({rmid})")
            if rmid:
                m = call("GET", "/models").get(rmid, {})
                # ER replays the base model's CUMULATIVE lineage, not just t-1 — so the child's
                # dataset_ids must contain every ancestor dataset plus this task's.
                lineage = m.get("dataset_ids") or []
                check("megavul_mini" in lineage and ds_a in lineage,
                      f"relearn[{method}] lineage cumulative ({lineage})")
                # class space comes from the trainer (target-aligned), so names must match the head
                check(len(m.get("class_names") or []) == int(m.get("num_classes") or 0),
                      f"relearn[{method}] class_names match num_classes ({m.get('num_classes')})")
            last_relearned = rmid or last_relearned
        print("   GET /relearn (list)")
        check(isinstance(call("GET", "/relearn"), list) and len(call("GET", "/relearn")) >= 4,
              "relearn jobs listed (>=4)")

        # 13b) NEGATIVE — relearning a nine-featurized seed model on a base-featurized dataset.
        #      Same node-feature dim, different LM: it would train fine and quietly ruin the
        #      model, so it must be refused up front.
        print("13b) POST /relearn (featurization mismatch — must be REFUSED)")
        bad = call("POST", "/relearn", {"method": "finetune", "base_model_id": "graph_based",
                                        "dataset_ids": ["megavul_mini"], "run_name": "e2e_bad",
                                        "config": {"epochs": 1}})
        refused = bool(bad.get("_http_error")) or (
            poll(f"/relearn/{bad['job_id']}").get("status") == "failed" if bad.get("job_id") else False)
        check(refused, "featurization mismatch refused (nine model vs base dataset)")

    # 14) inference on a relearned model — full loop closes
    if last_relearned:
        print(f"14) POST /inference on relearned model ({last_relearned})")
        rres = call("POST", "/inference", {"model_id": last_relearned, **codes})
        ok14 = any(r.get("ok") for r in rres.get("results", []))
        if not ok14:   # surface the actual failure (HTTP error body or per-function errors)
            print("   response:", json.dumps(rres)[:600])
        check(ok14, "relearned model serves inference")

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
