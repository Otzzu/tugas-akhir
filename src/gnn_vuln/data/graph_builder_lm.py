"""
graph_builder_lm.py
~~~~~~~~~~~~~~~~~~~
Builds PyG Data objects from Joern JSON CPGs using CodeBERT node embeddings.

Node feature vector (773D):
    [node_type_idx (1)] + [CodeBERT CLS token (768)] + [dist_features (3)] + [danger_api (1)]
Edge feature vector (7D one-hot):
    {AST, CFG, CDG, DDG, PDG, CALL, REACHING_DEF}

This module is only used during dataset preprocessing (CodeBERTGraphDataset.process).
The resulting Data objects are cached to disk so CodeBERT is only run once.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)
_unknown_enum_seen: set[str] = set()  # track unknown enum values, warn once per value

import torch
from torch_geometric.data import Data

from gnn_vuln.data.node_embedder import (
    CODEBERT_DIM,
    NODE_FEAT_DIM,
    NODE_TYPES,
    NODE_TYPE_TO_IDX,
    LMNodeEmbedder,
)

# All Joern/MegaVul edge types from SPECIFICATION.md + IMPORTS (Java graphs)
EDGE_TYPES = [
    "ARGUMENT",       # call → arguments
    "AST",            # abstract syntax tree
    "BINDS",          # type binding
    "CALL",           # function calls
    "CDG",            # control dependence
    "CFG",            # control flow
    "CONDITION",      # control structure → condition expr
    "CONTAINS",       # containment
    "DOMINATE",       # dominance
    "EVAL_TYPE",      # type evaluation
    "IMPORTS",        # import/include (Java/Python)
    "PARAMETER_LINK", # call arg → formal param
    "POST_DOMINATE",  # post-dominance
    "REACHING_DEF",   # data dependence (DDG)
    "RECEIVER",       # method call receiver
    "REF",            # identifier → declaration
    "SOURCE_FILE",    # source file link
]
EDGE_TYPE_TO_IDX = {et: i for i, et in enumerate(EDGE_TYPES)}

_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"

# Known dangerous C/C++ APIs — used as a binary node feature
DANGEROUS_APIS = frozenset({
    "malloc", "free", "realloc", "calloc",
    "memcpy", "memmove", "memset", "memcmp",
    "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strlen",
    "sprintf", "snprintf", "vsprintf", "vsnprintf",
    "gets", "fgets", "scanf", "sscanf",
    "read", "write", "recv", "send",
    "open", "fopen", "popen",
    "system", "execl", "execv", "execve",
    "alloca",
})

_TOKEN_RE = re.compile(r'\b\w+\b')


def _parse_graphml(path: str | Path) -> dict:
    """
    Parse a Joern GraphML (.xml) export into the same dict structure
    that build_lm_graph_from_json expects:
        {"nodes": [{id, labelV, code, lineNumber, ...}], "edges": [{src, dst, label}]}

    Joern stores attributes with uppercase names (CODE, LINE_NUMBER, labelE).
    We normalise them here so the rest of graph_builder_lm.py is unchanged.
    """
    def _tag(name: str) -> str:
        return f"{{{_GRAPHML_NS}}}{name}"

    tree = ET.parse(path)
    root = tree.getroot()

    # key id -> attr.name  (e.g. "node__METHOD__CODE" -> "CODE")
    key_map: dict[str, str] = {}
    for key_el in root.iter(_tag("key")):
        kid = key_el.get("id", "")
        aname = key_el.get("attr.name", kid)
        key_map[kid] = aname

    nodes: list[dict] = []
    edges: list[dict] = []

    for graph_el in root.iter(_tag("graph")):
        for node_el in graph_el.findall(_tag("node")):
            nd: dict = {"id": node_el.get("id")}
            for data_el in node_el.findall(_tag("data")):
                kid = data_el.get("key", "")
                aname = key_map.get(kid, kid)
                val = data_el.text or ""
                # Normalise to what graph_builder_lm expects
                if aname == "labelV":
                    nd["labelV"] = val
                elif aname == "CODE":
                    nd["code"] = val
                elif aname == "LINE_NUMBER":
                    nd["lineNumber"] = val
                else:
                    nd[aname] = val
            nodes.append(nd)

        for edge_el in graph_el.findall(_tag("edge")):
            ed: dict = {
                "src": edge_el.get("source"),
                "dst": edge_el.get("target"),
            }
            for data_el in edge_el.findall(_tag("data")):
                kid = data_el.get("key", "")
                aname = key_map.get(kid, kid)
                val = data_el.text or ""
                # labelE -> label so _edge_attr() finds it
                ed["label" if aname == "labelE" else aname] = val
            edges.append(ed)

    return {"nodes": nodes, "edges": edges}


_NUM_NODE_TYPES = len(NODE_TYPES)  # 22


def _node_type_onehot(attrs: dict) -> list[float]:
    """22D one-hot vector for node label. Unknown → UNKNOWN bucket."""
    label = str(attrs.get("labelV", attrs.get("label", "UNKNOWN")))
    if label not in NODE_TYPE_TO_IDX:
        _check_enum("node_label", label, set(NODE_TYPE_TO_IDX))
    idx = NODE_TYPE_TO_IDX.get(label, NODE_TYPE_TO_IDX["UNKNOWN"])
    vec = [0.0] * _NUM_NODE_TYPES
    vec[idx] = 1.0
    return vec


def _edge_attr(edge_attrs: dict) -> list[float]:
    etype = edge_attrs.get("label", "")
    if etype not in EDGE_TYPE_TO_IDX:
        _check_enum("edge_type", etype, set(EDGE_TYPE_TO_IDX))
        one_hot = [0.0] * len(EDGE_TYPES)
    else:
        one_hot = [0.0] * len(EDGE_TYPES)
        one_hot[EDGE_TYPE_TO_IDX[etype]] = 1.0
    # has_variable: 1 if REACHING_DEF edge carries a tracked variable name
    # GraphML stores it as "property"; MegaVul JSON stores it as "variable"
    var = edge_attrs.get("variable", edge_attrs.get("VARIABLE", edge_attrs.get("property", "")))
    has_var = 1.0 if var and str(var).strip() else 0.0
    return one_hot + [has_var]  # 18D total


def _node_line(attrs: dict) -> int:
    """Extract source line number from a CPG node; -1 if absent."""
    ln = attrs.get("lineNumber", attrs.get("LINE_NUMBER", None))
    try:
        return int(ln) if ln is not None else -1
    except (ValueError, TypeError):
        return -1


def _bfs_distances(
    sources: list[str],
    adjacency: dict[str, list[str]],
) -> dict[str, int]:
    """Multi-source BFS on node-id graph. Returns {node_id: hop_distance}."""
    dist: dict[str, int] = {}
    queue: deque = deque()
    for s in sources:
        if s not in dist:
            dist[s] = 0
            queue.append(s)
    while queue:
        curr = queue.popleft()
        for nb in adjacency.get(curr, []):
            if nb not in dist:
                dist[nb] = dist[curr] + 1
                queue.append(nb)
    return dist


def _compute_distance_features(
    nodes: list[dict],
    edges: list[dict],
) -> torch.Tensor:
    """
    Compute 3 VulChecker-inspired structural distance features per node,
    all normalised to [0, 1] by dividing by (N-1):

        [0] distance_from_entry      hops from METHOD node (forward edges)
        [1] distance_to_exit         hops to RETURN/METHOD_RETURN (reverse edges)
        [2] distance_to_nearest_call hops to nearest CALL node (reverse edges)

    Unreachable nodes get distance 1.0 (clamped max).
    Returns (N, 3) float32 tensor.
    """
    n = len(nodes)
    norm = max(n - 1, 1)

    adj:  dict[str, list[str]] = {nd["id"]: [] for nd in nodes}
    radj: dict[str, list[str]] = {nd["id"]: [] for nd in nodes}
    for e in edges:
        s, d = e.get("src"), e.get("dst")
        if s in adj and d in adj:
            adj[s].append(d)
            radj[d].append(s)

    def _label(nd: dict) -> str:
        return str(nd.get("labelV", nd.get("label", "UNKNOWN"))).upper()

    entry_ids = [nd["id"] for nd in nodes if _label(nd) == "METHOD"] or [nodes[0]["id"]]
    exit_ids  = [nd["id"] for nd in nodes if _label(nd) in ("RETURN", "METHOD_RETURN")]
    call_ids  = [nd["id"] for nd in nodes if _label(nd) == "CALL"]

    d_entry = _bfs_distances(entry_ids, adj)
    d_exit  = _bfs_distances(exit_ids,  radj)
    d_call  = _bfs_distances(call_ids,  radj)

    feats = [
        [
            min(d_entry.get(nd["id"], n) / norm, 1.0),
            min(d_exit.get(nd["id"],  n) / norm, 1.0),
            min(d_call.get(nd["id"],  n) / norm, 1.0),
        ]
        for nd in nodes
    ]
    return torch.tensor(feats, dtype=torch.float)  # (N, 3)


_CST_TO_IDX = {
    "IF": 1, "ELSE": 2, "WHILE": 3, "FOR": 4,
    "SWITCH": 5, "BREAK": 6, "CONTINUE": 7, "GOTO": 8,
    "TRY": 9, "CATCH": 10, "FINALLY": 11,  # Java exception handling
}
# 0 = missing/not-applicable, positive = actual value (ensures 0 is unambiguous)
_EVAL_TO_IDX  = {"BY_VALUE": 1, "BY_REFERENCE": 2, "BY_SHARING": 3}  # max=3
_DISPATCH_IDX = {"STATIC_DISPATCH": 1, "DYNAMIC_DISPATCH": 2}         # max=2
_DISPATCH_VALS = set(_DISPATCH_IDX)


def _check_enum(field: str, value: str, known: set | dict) -> None:
    """Warn once when an unknown enum value is encountered."""
    known_set = set(known) if not isinstance(known, set) else known
    if value and value not in known_set:
        key = f"{field}:{value}"
        if key not in _unknown_enum_seen:
            _unknown_enum_seen.add(key)
            _log.warning(
                "Unknown enum value for %s: %r — not in known set %s. "
                "Add it to the mapping in graph_builder_lm.py.",
                field, value, sorted(known_set)
            )
_INTEGER_TYPES = frozenset({
    "int", "long", "short", "char",
    "unsigned int", "unsigned long", "unsigned short", "unsigned char",
    "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
})
_MAX_ARG_IDX = 16.0  # normalisation cap; shifted by 1 so -1→0.0, 0→1/17, 1→2/17, ..., 16→1.0


def _dangerous_api_flags(nodes: list[dict], codes: list[str]) -> torch.Tensor:
    """(N,1): 1 if node code or methodFullName matches a known dangerous API."""
    flags = []
    for node, code in zip(nodes, codes):
        code_hit = any(t in DANGEROUS_APIS for t in _TOKEN_RE.findall(code))
        mfn = str(node.get("methodFullName", node.get("METHOD_FULL_NAME", "")))
        mfn_name = mfn.split(".")[-1].split(":")[0].split("<")[0]
        mfn_hit = mfn_name in DANGEROUS_APIS
        flags.append(1.0 if code_hit or mfn_hit else 0.0)
    return torch.tensor(flags, dtype=torch.float).unsqueeze(1)  # (N, 1)


def _extra_node_features(nodes: list[dict]) -> torch.Tensor:
    """
    (N, 27) float32 tensor of additional CPG node attributes:
      [0]    is_external         — binary: isExternal flag (library/API call)
      [1-12] ctrl_struct_type   — 12D one-hot: [none, IF, ELSE, WHILE, FOR, SWITCH, BREAK, CONTINUE, GOTO, TRY, CATCH, FINALLY]
      [13]   has_type_info       — binary: 1 if typeFullName present and non-empty
      [14]   is_pointer_type     — binary: typeFullName contains pointer
      [15]   is_integer_type     — binary: typeFullName is known integer type
      [16]   is_void_type        — binary: typeFullName is void/ANY
      [17-20] evaluation_strategy — 4D one-hot: [missing, BY_VALUE, BY_REFERENCE, BY_SHARING]
      [21]   argument_index      — continuous [0,1]: (raw+1)/17 shift; 0=missing/-1
      [22-24] dispatch_type      — 3D one-hot: [missing, STATIC_DISPATCH, DYNAMIC_DISPATCH]
      [25]   is_variadic         — binary: variadic parameter (..., va_list)
      [26]   span_normalized     — continuous [0,1]: (lineNumberEnd-lineNumber)/50, capped at 1
    """
    rows = []
    for n in nodes:
        # isExternal
        is_ext = 1.0 if str(n.get("isExternal", n.get("IS_EXTERNAL", "false"))).lower() == "true" else 0.0

        # controlStructureType — 12D one-hot [none, IF, ELSE, WHILE, FOR, SWITCH, BREAK, CONTINUE, GOTO, TRY, CATCH, FINALLY]
        cst = str(n.get("controlStructureType", n.get("CONTROL_STRUCTURE_TYPE", ""))).upper()
        if cst:
            _check_enum("controlStructureType", cst, _CST_TO_IDX)
        cst_vec = [0.0] * 12
        cst_vec[_CST_TO_IDX.get(cst, 0)] = 1.0

        # typeFullName features — has_type disambiguates missing from "other type"
        tfn = str(n.get("typeFullName", n.get("TYPE_FULL_NAME", ""))).strip()
        if tfn:
            has_type = 1.0
            is_ptr  = 1.0 if ("*" in tfn or "[]" in tfn) else 0.0
            is_int  = 1.0 if tfn.lower() in _INTEGER_TYPES else 0.0
            is_void = 1.0 if tfn.lower() in ("void", "any") else 0.0
        else:
            has_type = is_ptr = is_int = is_void = 0.0  # missing → no signal

        # evaluationStrategy — 4D one-hot [missing, BY_VALUE, BY_REFERENCE, BY_SHARING]
        es = str(n.get("evaluationStrategy", n.get("EVALUATION_STRATEGY", ""))).upper()
        if es:
            _check_enum("evaluationStrategy", es, _EVAL_TO_IDX)
        es_vec = [0.0] * 4
        es_vec[_EVAL_TO_IDX.get(es, 0)] = 1.0

        # argumentIndex — shift by 1: -1→0.0, 0→1/17, 1→2/17, ..., 16→1.0
        try:
            _raw = n.get("argumentIndex", n.get("ARGUMENT_INDEX", None))
            raw_ai = float(_raw) if _raw is not None else -1.0
            raw_ai = max(-1.0, min(raw_ai, _MAX_ARG_IDX))
            ai = (raw_ai + 1.0) / (_MAX_ARG_IDX + 1.0)
        except (ValueError, TypeError):
            ai = 0.0

        # dispatchType — 3D one-hot [missing, STATIC_DISPATCH, DYNAMIC_DISPATCH]
        dt = str(n.get("dispatchType", n.get("DISPATCH_TYPE", ""))).upper()
        if dt:
            _check_enum("dispatchType", dt, _DISPATCH_VALS)
        dt_vec = [0.0] * 3
        dt_vec[_DISPATCH_IDX.get(dt, 0)] = 1.0

        # isVariadic — binary: marks variadic params (..., va_list) → vararg/format-string vulns
        is_var = 1.0 if str(n.get("isVariadic", n.get("IS_VARIADIC", "false"))).lower() == "true" else 0.0

        # span_normalized — multi-line node size: 0=single-line/missing, normalized by cap
        try:
            ln_start = int(n.get("lineNumber", n.get("LINE_NUMBER", 0)) or 0)
            ln_end   = int(n.get("lineNumberEnd", n.get("LINE_NUMBER_END", ln_start)) or ln_start)
            span = max(0, ln_end - ln_start)
            span_norm = min(span, 50) / 50.0  # cap at 50 lines
        except (ValueError, TypeError):
            span_norm = 0.0

        rows.append([is_ext] + cst_vec + [has_type, is_ptr, is_int, is_void] + es_vec + [ai] + dt_vec + [is_var, span_norm])

    return torch.tensor(rows, dtype=torch.float)  # (N, 27)


def parse_cpg(path: str | Path, max_nodes: int = 500) -> Optional[dict]:
    """
    Parse a Joern CPG file (.json or .xml/.graphml) and validate its size.

    Returns a dict with keys {nodes, edges, codes} ready for build_from_parsed(),
    or None if the file is empty or exceeds max_nodes.
    Separating parsing from embedding lets the caller batch-embed all graphs at once.
    """
    path = Path(path)
    if path.suffix.lower() in (".xml", ".graphml"):
        cpg = _parse_graphml(path)
    else:
        with open(path, encoding="utf-8") as f:
            cpg = json.load(f)

    nodes = cpg.get("nodes", [])
    edges = cpg.get("edges", [])

    if not nodes or len(nodes) > max_nodes:
        return None

    cpg["codes"] = [str(n.get("code", "")) for n in nodes]
    return cpg


def build_func_text(cpg: dict) -> str:
    """
    Reconstruct a readable function body from CPG node code snippets,
    sorted by source line number. Used to tokenize the full function for
    live CodeBERT injection (Architecture 3 / 4).
    """
    nodes = cpg["nodes"]
    codes = cpg["codes"]
    line_codes: dict[int, str] = {}
    for code, node in zip(codes, nodes):
        line = _node_line(node)
        if line >= 0 and code.strip() and line not in line_codes:
            line_codes[line] = code
    if line_codes:
        return "\n".join(line_codes[l] for l in sorted(line_codes))
    # Fallback: join first 50 code snippets (no line info available)
    return " ".join(c for c in codes[:50] if c.strip()) or "<empty>"


def build_from_parsed(
    cpg: dict,
    cls_feats: torch.Tensor,
    label: int,
    flaw_lines: list[int] | None = None,
    func_input_ids: torch.Tensor | None = None,
    func_attention_mask: torch.Tensor | None = None,
) -> Data:
    """
    Build a PyG Data object from a pre-parsed CPG and pre-computed CodeBERT embeddings.

    Parameters
    ----------
    cpg       : dict returned by parse_cpg() — contains nodes, edges, codes
    cls_feats : (N, 768) CodeBERT CLS embeddings for each node, already on CPU
    label     : graph-level class label
    flaw_lines: 1-indexed known vulnerable line numbers (empty → all-zero mask)

    Returns
    -------
    Data with x=[N, NON_LM_FEAT_DIM+lm_dim], edge_index=[2,E], edge_attr=[E,18], y=[1],
    node_line=[N], flaw_line_mask=[N]
    """
    nodes = cpg["nodes"]
    edges = cpg["edges"]
    codes = cpg["codes"]

    node_ids = [n["id"] for n in nodes]
    node_idx = {nid: i for i, nid in enumerate(node_ids)}

    type_oh      = torch.tensor([_node_type_onehot(n) for n in nodes], dtype=torch.float)  # (N, 22)
    dist_feats   = _compute_distance_features(nodes, edges)                                 # (N, 3)
    danger_flags = _dangerous_api_flags(nodes, codes)                                       # (N, 1)
    extra_feats  = _extra_node_features(nodes)                                              # (N, 27)

    x = torch.cat([type_oh, cls_feats, dist_feats, danger_flags, extra_feats], dim=1)

    node_lines = [_node_line(n) for n in nodes]
    node_line  = torch.tensor(node_lines, dtype=torch.long)
    flaw_set   = set(flaw_lines) if flaw_lines else set()

    # flaw_line_mask: covers full node span [lineNumber, lineNumberEnd]
    def _node_end_line(nd: dict) -> int:
        try:
            v = nd.get("lineNumberEnd", nd.get("LINE_NUMBER_END", None))
            return int(v) if v is not None else -1
        except (ValueError, TypeError):
            return -1

    flaw_line_mask = torch.tensor(
        [1 if (ln >= 0 and any(ln <= fl <= max(ln, _node_end_line(n)) for fl in flaw_set))
         else 0
         for ln, n in zip(node_lines, nodes)],
        dtype=torch.long,
    )

    src_list, dst_list, edge_attr_list = [], [], []
    for e in edges:
        src, dst = e.get("src"), e.get("dst")
        if src in node_idx and dst in node_idx:
            src_list.append(node_idx[src])
            dst_list.append(node_idx[dst])
            edge_attr_list.append(_edge_attr(e))

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, len(EDGE_TYPES)), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
        node_line=node_line,
        flaw_line_mask=flaw_line_mask,
    )
    if func_input_ids is not None:
        data.func_input_ids = func_input_ids        # [1, 512] — batches to [B, 512]
        data.func_attention_mask = func_attention_mask  # [1, 512]
    return data


def build_lm_graph_from_json(
    path: str | Path,
    label: int,
    embedder: LMNodeEmbedder,
    max_nodes: int = 500,
    embed_batch_size: int = 256,
    flaw_lines: list[int] | None = None,
) -> Optional[Data]:
    """Convenience wrapper: parse + embed + build in one call (single-graph use)."""
    cpg = parse_cpg(path, max_nodes)
    if cpg is None:
        return None
    codes = cpg["codes"]
    cls_parts = [
        embedder.embed_batch(codes[i : i + embed_batch_size])
        for i in range(0, len(codes), embed_batch_size)
    ]
    cls_feats = torch.cat(cls_parts, dim=0)
    return build_from_parsed(cpg, cls_feats, label, flaw_lines)
