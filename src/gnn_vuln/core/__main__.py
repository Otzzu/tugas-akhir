"""CLI di atas core — satu subprocess menggantikan dua (prepare + build_pt), dan hasilnya
dikembalikan sebagai JSON, bukan dititipkan lewat berkas internal library.

Subprocess tetap dipakai bukan karena antarmukanya CLI, tapi karena proses yang mati
membebaskan memori GPU dengan bersih.

    python -m gnn_vuln.core build-dataset \
        --rows rows.parquet --out data/processed/ds_x.pt \
        --pretrained-lm microsoft/unixcoder-base-nine --joern-cli /opt/joern/joern-cli \
        --result-json out/info.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gnn_vuln.core import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser(prog="gnn_vuln.core")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dataset", help="rows -> CPG -> .pt, at the path you name")
    b.add_argument("--rows", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path, help="<root>/processed/<name>.pt")
    b.add_argument("--pretrained-lm", required=True)
    b.add_argument("--func-lm", default="")
    b.add_argument("--func-max-length", type=int, default=1024)
    b.add_argument("--no-func-tokens", action="store_true")
    b.add_argument("--mode", default="multiclass")
    b.add_argument("--storage", default="inmemory", choices=["inmemory", "lazy"])
    b.add_argument("--max-nodes", type=int, default=2500)
    b.add_argument("--joern-cli", type=Path)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--device", default="cpu")
    b.add_argument("--cwe-vocab-json", type=Path,
                   help="reuse a parent dataset's class space (a JSON {cwe: id})")
    b.add_argument("--result-json", type=Path, help="where to write the DatasetInfo")

    m = sub.add_parser("merge-datasets", help="merge built .pt datasets into one, by path")
    m.add_argument("--paths", required=True, nargs="+", type=Path)
    m.add_argument("--out", required=True, type=Path)
    m.add_argument("--no-dedup", action="store_true")
    m.add_argument("--storage", default="inmemory", choices=["inmemory", "lazy"])
    m.add_argument("--device", default="cpu")
    m.add_argument("--result-json", type=Path)

    args = ap.parse_args()

    if args.cmd == "merge-datasets":
        from gnn_vuln.core import merge_datasets
        info = merge_datasets(args.paths, args.out, dedup=not args.no_dedup,
                              storage=args.storage, device=args.device)
        payload = asdict(info) | {"path": str(info.path)}
        if args.result_json:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            args.result_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    vocab = json.loads(args.cwe_vocab_json.read_text(encoding="utf-8")) \
        if args.cwe_vocab_json else None

    info = build_dataset(
        rows=args.rows, out_path=args.out,
        featurization={
            "pretrained_lm": args.pretrained_lm,
            "func_lm": args.func_lm or args.pretrained_lm,
            "add_func_tokens": not args.no_func_tokens,
            "func_max_length": args.func_max_length,
        },
        joern_cli=args.joern_cli, cwe_vocab=vocab, max_nodes=args.max_nodes,
        mode=args.mode, storage=args.storage, workers=args.workers, device=args.device,
    )

    payload = asdict(info) | {"path": str(info.path)}
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
