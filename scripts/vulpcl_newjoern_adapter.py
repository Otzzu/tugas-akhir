"""
vulpcl_newjoern_adapter.py — build VulPCL categorization pkls from OUR megavul, using MODERN joern
CPGs (joern 4.0.526) instead of VulPCL's dead 2014 Neo4j/SVG pipeline. Produces the same 3 reps +
pkl format VulPCL's CodeBert_Blstm consumes, on our split + classes -> comparable head-to-head.

FAITHFULNESS: the MODEL (CodeBert_Blstm) is untouched. Only the graph->features step is re-derived
(their SVG/old-joern extractors can't read modern joern). Features APPROXIMATE VulPCL's — documented.

Each pkl item = [id, codebert_ids(512), fcds_ids(512), cpag_deepwalk(256x300 float), label_str]:
  PTSC  -> codebert_ids: CodeBERT tokenizer on source, 512 (pad/trunc).
  FCDS  -> fcds_ids: code tokens of statement-ish nodes (have code + lineNumber), ordered by line,
           tokenized (VulPCL-style splitter) -> FCDS vocab ids, 512. Vocab built from TRAIN.
  CPAG  -> cpag_deepwalk: GLOBAL graph over code-token nodes (edges = AST+DDG+CDG between them, all
           funcs) -> DeepWalk (walk_length=10, num_walks=80, dim=300, window=5, iter=3, matches adc)
           -> per-code embedding. Per-func = its nodes' embeddings, padded/trunc to 256 -> [256,300].

Inputs:
  --split-dir : megavul_ml1024/linevd  (train/val/test.parquet: func_before, vul, cwe_name, id)
  --cpg-dir   : dir with func_<id>.json (+ .meta.json) modern-joern CPGs for those funcs
  --out-dir   : writes {train,val,test}_set_token.pkl + fcds_code_vocab.json + cwe_labels.json
  --vulpcl    : path to src/VulPCL (to import deepwalk_embed)
Run: python scripts/vulpcl_newjoern_adapter.py --split-dir megavul_ml1024/linevd \
        --cpg-dir megavul_vulpcl_cpg --out-dir megavul_vulpcl_pkl --vulpcl src/VulPCL
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CB_MAX, FCDS_MAX, CPAG_NODES, DW_DIM = 512, 512, 256, 300
FCDS_EDGE_LABELS = {"AST", "DDG", "CDG", "CFG", "REACHING_DEF"}   # composite for CPAG global graph
SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3, "<MASK>": 4}
# statement-ish modern-joern node labels carrying code (approx VulPCL's key_ntypes statement set)
STMT_LABELS = {"CALL", "CONTROL_STRUCTURE", "RETURN", "IDENTIFIER", "LITERAL", "BLOCK",
               "JUMP_TARGET", "LOCAL", "METHOD_PARAMETER_IN", "FIELD_IDENTIFIER", "UNKNOWN"}

_SPLIT = re.compile(r'\"(.*?)\"| +|(;)|(->)|(&)|(\*)|(\()|(==)|(~)|(!=)|(<=)|(>=)|(!)|(\+\+)|(--)'
                    r'|(\))|(=)|(\+)|(\-)|(\[)|(\])|(<)|(>)|(\.)|({)')


def tok(code: str) -> list[str]:
    """VulPCL-style code tokenizer (mirrors their splitter)."""
    parts = re.split(_SPLIT, code or "")
    return [p for p in parts if p and p.strip() and p not in ("{", "}", ";", ":")]


def load_cpg(p):
    """Return (nodes: {id:(code,label,line)}, edges: [(src,dst,etype)]) or None."""
    if p is None or not Path(p).exists():
        return None
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None
    nodes, edges = {}, []
    for n in d.get("nodes", []):
        nodes[n.get("id")] = (str(n.get("code", "")), n.get("_label", ""), n.get("lineNumber"))
    for e in d.get("edges", []):
        edges.append((e.get("outNode"), e.get("inNode"), e.get("etype", "")))
    return nodes, edges


def fcds_tokens(nodes) -> list[str]:
    """Statement-ish nodes with code, ordered by line, code tokens concatenated (FCDS)."""
    sel = [(ln if ln is not None else 1e9, code) for (code, lab, ln) in nodes.values()
           if code and lab in STMT_LABELS]
    sel.sort(key=lambda x: x[0])
    out: list[str] = []
    for _, code in sel:
        out.extend(tok(code))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--cpg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vulpcl", default="src/VulPCL")
    a = ap.parse_args()
    split_dir, cpg_dir, out = Path(a.split_dir), Path(a.cpg_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(a.vulpcl) / "vul_categorization"))
    from deepwalk_embed.deepwalk_embedding import deepwalk  # noqa: their DeepWalk
    import networkx as nx
    from transformers import AutoTokenizer
    cb_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # CPGs come from prepare_dataset (func_<localidx>.json + .meta.json with raw_func). prepare_dataset
    # renames files by local index, so match CPG <-> function by SOURCE TEXT (whitespace-normalized).
    def wskey(s: str) -> str:
        return "".join(str(s).split())
    src2cpg: dict[str, Path] = {}
    for mp in cpg_dir.rglob("*.meta.json"):     # prepare_dataset writes benign/ + vulnerable/ subdirs
        try:
            rf = json.loads(mp.read_text(encoding="utf-8")).get("raw_func", "")
        except Exception:
            continue
        jp = mp.with_name(mp.name.replace(".meta.json", ".json"))
        if rf and jp.exists():
            src2cpg[wskey(rf)] = jp
    print(f"  CPGs indexed by source: {len(src2cpg)}")

    # ---- gather funcs per split (26-class: benign=0, 25 CWE = 1..25) ----
    frames = {}
    for s in ("train", "val", "test"):
        df = pd.read_parquet(split_dir / f"{s}.parquet")
        frames[s] = df
    allcwe = pd.concat(frames.values())
    allcwe["cwe_name"] = allcwe["cwe_name"].fillna("others").replace("", "others").astype(str)
    allcwe.loc[allcwe["vul"] == 0, "cwe_name"] = "benign"
    vuln = sorted(allcwe[allcwe["vul"] == 1]["cwe_name"].unique())
    cwe_labels = ["benign"] + vuln
    c2l = {c: i for i, c in enumerate(cwe_labels)}
    (out / "cwe_labels.json").write_text(json.dumps({"cwe_labels": cwe_labels, "num_classes": len(cwe_labels)}, indent=2))

    # ---- pass 1: load CPGs, FCDS tokens, global CPAG edgelist (node = code string) ----
    recs = {s: [] for s in ("train", "val", "test")}     # (id, source, label, fcds_toks, code_nodes)
    fcds_counter: Counter = Counter()
    G = nx.DiGraph()
    for s in ("train", "val", "test"):
        df = frames[s]
        for _, r in df.iterrows():
            fid = int(r["id"])
            cpg = load_cpg(src2cpg.get(wskey(r["func_before"])))
            if cpg is None:
                continue
            nodes, edges = cpg
            cn = "benign" if r["vul"] == 0 else (str(r["cwe_name"]) if pd.notna(r["cwe_name"]) and r["cwe_name"] else "others")
            label = c2l.get(cn, 0)
            ft = fcds_tokens(nodes)
            if s == "train":
                fcds_counter.update(ft)
            # global CPAG: code-string nodes + composite edges
            codenodes = []
            for src, dst, et in edges:
                if et in FCDS_EDGE_LABELS and src in nodes and dst in nodes:
                    cs, cd = nodes[src][0].strip(), nodes[dst][0].strip()
                    if cs and cd:
                        G.add_edge(cs, cd)
            for (code, lab, ln) in nodes.values():
                if code.strip() and lab in STMT_LABELS:
                    codenodes.append(code.strip())
            recs[s].append((fid, str(r["func_before"]), label, ft, codenodes))

    # ---- FCDS vocab from train ----
    vocab = dict(SPECIAL)
    for i, (w, _) in enumerate(fcds_counter.most_common()):
        vocab[w] = i + len(SPECIAL)
    (out / "fcds_code_vocab.json").write_text(json.dumps(vocab))

    # ---- global DeepWalk over CPAG graph (matches adc params) ----
    emb: dict[str, np.ndarray] = {}
    if G.number_of_nodes() > 0:
        dp = deepwalk(G, walk_length=10, num_walks=80, workers=1)
        dp.train(window_size=5, iter=3)        # embed_size=300 default
        emb = dp.get_embedding()
    zero = np.zeros(DW_DIM, dtype=np.float32)

    def cb_ids(src: str) -> list[int]:
        e = cb_tok(src, max_length=CB_MAX, truncation=True, padding="max_length")
        return e["input_ids"]

    def fcds_ids(toks: list[str]) -> list[int]:
        ids = [vocab.get(t, SPECIAL["<UNK>"]) for t in toks][:FCDS_MAX]
        return ids + [SPECIAL["<PAD>"]] * (FCDS_MAX - len(ids))

    def cpag_mat(codenodes: list[str]) -> list[list[float]]:
        rows = []
        for c in codenodes[:CPAG_NODES]:
            v = emb.get(c)
            rows.append((np.asarray(v, dtype=np.float32) if v is not None else zero).tolist())
        while len(rows) < CPAG_NODES:
            rows.append(zero.tolist())
        return rows

    # ---- pass 2: write pkls ----
    for s in ("train", "val", "test"):
        items = []
        for fid, src, label, ft, codenodes in recs[s]:
            items.append([f"{fid}@megavul", cb_ids(src), fcds_ids(ft), cpag_mat(codenodes), str(label)])
        pickle.dump(items, open(out / f"{s}_set_token.pkl", "wb"))
        print(f"  {s}: {len(items)} items")
    print(f"num_classes={len(cwe_labels)}  fcds_vocab={len(vocab)}  cpag_nodes_embedded={len(emb)}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
