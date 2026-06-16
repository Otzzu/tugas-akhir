"""
vulpcl_localization_adapter.py — build VulPCL *localization* pkls from OUR megavul, reusing our hdf5 CPGs
(joern 4.0.526) — NO joern re-parse. Mirrors the categorization adapter but for vul_localization/.

VulPCL localization = a BINARY (vuln/benign) CodeBert_Blstm classifier; line localization is done by
reading CodeBERT's attention (output_attentions) -> per-line score -> top-k line accuracy. So the data is
binary-labelled (benign INCLUDED) + carries vul_idx (vulnerable line indices) + a token-string seq used
to map attention positions -> lines (via the 'Ċ' newline + '</s>' tokens).

Per their dataset_iter, each token pkl item = [id_str("@<int>"), codebert512, fcds512, cpag[256,300],
label_binary, vul_idx(list[str])]. Plus a parallel <split>_seq.pkl item = [id_str, token_strings] and a
big_vul_msg.txt mapping id->CWE (for their top-10-CWE line_tp metric).

LINE-MAPPING [VERIFY on pod]: their atten_score_process counts lines by 'Ċ' tokens; line_num 0 = func line
1 (skipped, li==0 continue), line_num k = func line k+1. So func flaw line i (1-indexed) -> vul_idx str(i-1).
If localization numbers look off, revisit this off-by-one.

Run: python scripts/vulpcl_localization_adapter.py --hdf5 data/graphs/megavul.hdf5 \
        --split-dir megavul_ml1024/linevd --out-dir megavul_vulpcl_loc_pkl --vulpcl src/VulPCL
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

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


def _ln(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 10 ** 9


def flaw_set(v) -> list[int]:
    if v is None:
        return []
    try:
        return sorted({int(x) for x in list(v)})
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vulpcl", default="src/VulPCL")
    a = ap.parse_args()
    ind, out = Path(a.split_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(a.vulpcl) / "vul_localization"))
    from deepwalk_embed.deepwalk_embedding import deepwalk
    import networkx as nx
    from transformers import AutoTokenizer
    cb_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    # split: BOTH benign + vuln (binary). keep func_before (full, for tokenize+line-map), flaw, cwe.
    frames = {s: pd.read_parquet(ind / f"{s}.parquet") for s in ("train", "val", "test")}
    id2info: dict[int, tuple] = {}
    for s in ("train", "val", "test"):
        for _, r in frames[s].iterrows():
            rid = int(r["id"])
            cwe = str(r.get("cwe_name") or "").strip()
            id2info[rid] = (s, int(r["vul"]), flaw_set(r.get("flaw_lines")), str(r["func_before"]), cwe)

    # pass 1: hdf5 -> FCDS tokens + global CPAG graph (node = code string), matched by row_id
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
                split, vul, flaws, func_before, cwe = id2info[rid]
                nodes_list = json.loads(_decode(g["nodes_json"][()]))
                edges_list = json.loads(_decode(g["edges_json"][()]))
                nodes = {n.get("id"): (str(n.get("code", "")), n.get("labelV", ""), n.get("lineNumber"))
                         for n in nodes_list}
                sel = sorted(((_ln(ln), code)
                              for (code, lab, ln) in nodes.values() if code and lab in STMT_LABELS),
                             key=lambda x: x[0])
                ft = [t for _, code in sel for t in tok(code)]
                if split == "train":
                    fcds_counter.update(ft)
                for e in edges_list:
                    if e.get("label") in FCDS_EDGE_LABELS:
                        s_, d_ = e.get("src"), e.get("dst")
                        if s_ in nodes and d_ in nodes:
                            cs, cd = nodes[s_][0].strip(), nodes[d_][0].strip()
                            if cs and cd:
                                G.add_edge(cs, cd)
                codenodes = [code.strip() for (code, lab, ln) in nodes.values()
                             if code.strip() and lab in STMT_LABELS]
                recs[split].append((rid, func_before, vul, flaws, ft, codenodes, cwe))
                matched += 1
    print(f"  matched {matched} funcs from hdf5 to split")

    vocab = dict(SPECIAL)
    for i, (w, _) in enumerate(fcds_counter.most_common()):
        vocab[w] = i + len(SPECIAL)

    emb: dict[str, np.ndarray] = {}
    if G.number_of_nodes() > 0:
        dp = deepwalk(G, walk_length=10, num_walks=80, workers=1)
        dp.train(window_size=5, iter=3)
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

    msg_blocks = []
    counts = {s: [0, 0] for s in ("train", "val", "test")}    # [benign, vuln]
    for s in ("train", "val", "test"):
        token_items, seq_items = [], []
        for rid, func_before, vul, flaws, ft, codenodes, cwe in recs[s]:
            ids = cb_ids(func_before)
            # func line i (1-idx) -> their line_num (i-1); line_num 0 (func line 1) is skipped by their metric
            vul_idx = [str(i - 1) for i in flaws] if vul else []
            token_items.append([f"@{rid}", ids, fcds_ids(ft), cpag_mat(codenodes), str(int(vul)), vul_idx])
            seq_items.append([f"@{rid}", cb_tok.convert_ids_to_tokens(ids)])
            counts[s][int(vul)] += 1
            if vul and cwe:
                msg_blocks.append(f"@{rid}.c&&_&&{cwe}")
        pickle.dump(token_items, open(out / f"{s}_set_token.pkl", "wb"))
        pickle.dump(seq_items, open(out / f"{s}_seq.pkl", "wb"))
        print(f"  {s}: {len(token_items)} items (benign {counts[s][0]} / vuln {counts[s][1]})")

    (out / "megavul_code_vocab.json").write_text(json.dumps({str(i): i for i in range(len(vocab))}))
    (out / "big_vul_msg.txt").write_text("\n--------------------------\n".join(msg_blocks))
    print(f"num_classes=2(binary)  fcds_vocab={len(vocab)}  cpag_nodes_embedded={len(emb)} -> {out}")


if __name__ == "__main__":
    main()
