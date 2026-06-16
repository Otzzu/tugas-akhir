"""
vulpcl_newjoern_adapter.py — build VulPCL categorization pkls from OUR megavul, REUSING our existing
CPGs in data/graphs/megavul.hdf5 (joern 4.0.526, the SAME graphs our model uses) — NO joern re-parse.
Produces VulPCL's 3 reps + pkl format their CodeBert_Blstm consumes, on our split + vuln-only 25 classes
(no benign — matches VulPCL categorization §3.3.2 + LOSVER/LIVABLE fairness).

FAITHFULNESS: model (CodeBert_Blstm) untouched. Only graph->features re-derived from modern-joern CPGs
(their SVG/old-joern extractors can't read modern joern). Features APPROXIMATE VulPCL's — documented.

hdf5 layout: {benign,vulnerable}/func_X groups, each with nodes_json + edges_json (the CPG) and attrs
row_id, cwe, flaw_lines, raw_func, label. We match group.attrs['row_id'] to the split parquet id.

Each pkl item = [id, codebert_ids(512), fcds_ids(512), cpag_deepwalk(256x300 float), label_str]:
  PTSC -> codebert_ids: CodeBERT tokenizer on raw_func, 512.
  FCDS -> fcds_ids: code tokens of statement-ish nodes (have code + a stmt labelV), tokenized
          (VulPCL-style splitter) -> FCDS vocab ids (built from TRAIN), 512.
  CPAG -> cpag_deepwalk: GLOBAL graph over code-token nodes (edges AST+DDG+CDG+CFG, all funcs) ->
          DeepWalk (walk_length=10, num_walks=80, dim=300, window=5, iter=3, == adc) -> per-code
          embedding; per-func = its nodes' embeddings padded/trunc to 256 -> [256,300].

Run: python scripts/vulpcl_newjoern_adapter.py --hdf5 data/graphs/megavul.hdf5 \
        --split-dir megavul_ml1024/linevd --out-dir megavul_vulpcl_pkl --vulpcl src/VulPCL
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

# gensim Word2Vec logs per-epoch progress at INFO -> makes the DeepWalk phase trackable (not silent)
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

import h5py
import numpy as np
import pandas as pd

CB_MAX, FCDS_MAX, CPAG_NODES, DW_DIM = 512, 512, 256, 300
FCDS_EDGE_LABELS = {"AST", "DDG", "CDG", "CFG", "REACHING_DEF"}
SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3, "<MASK>": 4}
STMT_LABELS = {"CALL", "CONTROL_STRUCTURE", "RETURN", "IDENTIFIER", "LITERAL", "BLOCK",
               "JUMP_TARGET", "LOCAL", "METHOD_PARAMETER_IN", "FIELD_IDENTIFIER", "UNKNOWN"}
_SPLIT = re.compile(r'\"(.*?)\"| +|(;)|(->)|(&)|(\*)|(\()|(==)|(~)|(!=)|(<=)|(>=)|(!)|(\+\+)|(--)'
                    r'|(\))|(=)|(\+)|(\-)|(\[)|(\])|(<)|(>)|(\.)|({)')


def tok(code: str) -> list[str]:
    return [p for p in re.split(_SPLIT, code or "") if p and p.strip() and p not in ("{", "}", ";", ":")]


def _decode(v) -> str:
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def _ln(v) -> int:                       # hdf5 lineNumber may be str/int/None -> sortable int
    try:
        return int(v)
    except (TypeError, ValueError):
        return 10 ** 9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vulpcl", default="src/VulPCL")
    ap.add_argument("--num-walks", type=int, default=10,
                    help="DeepWalk walks/node (paper 80; 10 ample at our ~680k-node global graph, far faster)")
    a = ap.parse_args()
    ind, out = Path(a.split_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(a.vulpcl) / "vul_categorization"))
    from deepwalk_embed.deepwalk_embedding import deepwalk
    import networkx as nx
    from transformers import AutoTokenizer
    cb_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # VULN-ONLY label space (no benign) — matches VulPCL categorization (§3.3.2 relabels vuln funcs only)
    # + LOSVER/LIVABLE fairness (vuln-only 25-class). keyed by id (== hdf5 row_id).
    frames = {s: pd.read_parquet(ind / f"{s}.parquet") for s in ("train", "val", "test")}
    allcwe = pd.concat(frames.values())
    allcwe["cwe_name"] = allcwe["cwe_name"].astype(str).str.strip()
    vuln = sorted(c for c in allcwe[allcwe["vul"] == 1]["cwe_name"].unique() if c.startswith("CWE-"))
    cwe_labels = vuln                                   # 25 CWE, NO benign
    c2l = {c: i for i, c in enumerate(cwe_labels)}
    (out / "cwe_labels.json").write_text(json.dumps({"cwe_labels": cwe_labels, "num_classes": len(cwe_labels)}, indent=2))
    id2info: dict[int, tuple[str, int]] = {}
    for s in ("train", "val", "test"):
        for _, r in frames[s].iterrows():
            if int(r["vul"]) == 0:                      # drop benign
                continue
            cn = str(r["cwe_name"]).strip()
            if cn not in c2l:                           # drop NaN/others/non-top CWE
                continue
            id2info[int(r["id"])] = (s, c2l[cn])

    # pass 1: read hdf5, build FCDS tokens + global CPAG edgelist (node = code string)
    recs = {s: [] for s in ("train", "val", "test")}
    fcds_counter: Counter = Counter()
    G = nx.DiGraph()
    matched = 0
    with h5py.File(a.hdf5, "r") as h:
        for top in ("benign", "vulnerable"):
            if top not in h:
                continue
            for name in h[top]:
                g = h[top][name]
                try:
                    rid = int(g.attrs["row_id"])
                except Exception:
                    continue
                if rid not in id2info or "nodes_json" not in g or "edges_json" not in g:
                    continue
                split, label = id2info[rid]
                nodes_list = json.loads(_decode(g["nodes_json"][()]))
                edges_list = json.loads(_decode(g["edges_json"][()]))
                nodes = {n.get("id"): (str(n.get("code", "")), n.get("labelV", ""), n.get("lineNumber"))
                         for n in nodes_list}
                raw = _decode(g.attrs.get("raw_func", ""))
                # FCDS tokens (statement nodes, by line)
                sel = sorted(((_ln(ln), code)
                              for (code, lab, ln) in nodes.values() if code and lab in STMT_LABELS),
                             key=lambda x: x[0])
                ft = [t for _, code in sel for t in tok(code)]
                if split == "train":
                    fcds_counter.update(ft)
                # global CPAG edges (code-string nodes)
                for e in edges_list:
                    if e.get("label") in FCDS_EDGE_LABELS:        # hdf5 edge schema = {src,dst,label}
                        s_, d_ = e.get("src"), e.get("dst")
                        if s_ in nodes and d_ in nodes:
                            cs, cd = nodes[s_][0].strip(), nodes[d_][0].strip()
                            if cs and cd:
                                G.add_edge(cs, cd)
                codenodes = [code.strip() for (code, lab, ln) in nodes.values()
                             if code.strip() and lab in STMT_LABELS]
                recs[split].append((rid, raw, label, ft, codenodes))
                matched += 1
    print(f"  matched {matched} funcs from hdf5 to split")

    vocab = dict(SPECIAL)
    for i, (w, _) in enumerate(fcds_counter.most_common()):
        vocab[w] = i + len(SPECIAL)
    (out / "fcds_code_vocab.json").write_text(json.dumps(vocab))

    emb: dict[str, np.ndarray] = {}
    if G.number_of_nodes() > 0:
        nw = os.cpu_count() or 4                      # parallel walk-gen + w2v (was workers=1 = 1 core)
        print(f"  deepwalk on {G.number_of_nodes()} nodes / {G.number_of_edges()} edges, workers={nw} num_walks={a.num_walks}")
        dp = deepwalk(G, walk_length=10, num_walks=a.num_walks, workers=nw)
        dp.train(window_size=5, iter=3, workers=nw)
        emb = dp.get_embedding()
    zero = np.zeros(DW_DIM, dtype=np.float32)

    def cb_ids(src: str) -> list[int]:
        return cb_tok(src, max_length=CB_MAX, truncation=True, padding="max_length")["input_ids"]

    def fcds_ids(toks: list[str]) -> list[int]:
        ids = [vocab.get(t, SPECIAL["<UNK>"]) for t in toks][:FCDS_MAX]
        return ids + [SPECIAL["<PAD>"]] * (FCDS_MAX - len(ids))

    def cpag_mat(codenodes: list[str]) -> list[list[float]]:
        rows = [(np.asarray(emb[c], dtype=np.float32) if c in emb else zero).tolist()
                for c in codenodes[:CPAG_NODES]]
        rows += [zero.tolist()] * (CPAG_NODES - len(rows))
        return rows

    for s in ("train", "val", "test"):
        items = [[f"{rid}@megavul", cb_ids(raw), fcds_ids(ft), cpag_mat(cn), str(label)]
                 for rid, raw, label, ft, cn in recs[s]]
        pickle.dump(items, open(out / f"{s}_set_token.pkl", "wb"))
        print(f"  {s}: {len(items)} items")
    print(f"num_classes={len(cwe_labels)}  fcds_vocab={len(vocab)}  cpag_nodes_embedded={len(emb)} -> {out}")


if __name__ == "__main__":
    main()
