"""
analyze_cwe_tree.py
~~~~~~~~~~~~~~~~~~~
Parse CWE XML taxonomy tree to:
1. Build parent-child hierarchy
2. Find tree-based groups for all dataset CWEs
3. Generate distance matrix for hierarchical contrastive loss
4. Suggest improved CWE_GROUP_MAP based on actual CWE tree

Usage:
    uv run python scripts/analyze_cwe_tree.py --xml data/cwe/699.xml
    uv run python scripts/analyze_cwe_tree.py --xml data/cwe/699.xml --depth 2
    uv run python scripts/analyze_cwe_tree.py --xml data/cwe/699.xml --matrix --cwes 119 125 787 416 476
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path


# CWEs present in BigVul + DiverseVul + MegaVul datasets (excluding UNKNOWN/Other)
DATASET_CWES = [
    "119", "125", "20", "399", "200", "264", "189", "416", "190", "362",
    "476", "787", "284", "254", "310", "415", "732", "404", "19", "79",
    "77", "78", "94", "89", "74", "90", "835", "704", "617", "358", "388",
    "674", "834", "426", "532", "754", "755", "209", "287", "255", "346",
    "295", "401", "122", "908", "667", "863", "459", "681", "367", "843",
    "1284", "121", "326", "662", "276", "703", "613", "319", "697", "203",
    "212", "444", "552", "116", "131", "440", "911", "385", "407", "823",
    "670", "337", "126", "1077", "241", "1188", "665", "672", "1333", "434",
    "184", "400", "772", "770", "664", "362", "369", "682", "191", "129",
    "134", "120", "909", "763", "824", "494", "502", "1021", "311", "330",
    "347", "522", "327", "17", "18", "693", "836", "862", "918", "601",
    "281", "352", "285", "268", "269", "22", "59", "836",
]


def parse_xml(xml_path: str) -> tuple[dict, dict, dict]:
    """Parse CWE XML → parent_map, child_map, name_map."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    parent_map: dict[str, list[str]] = defaultdict(list)  # child → [parents]
    child_map: dict[str, list[str]] = defaultdict(list)   # parent → [children]
    name_map: dict[str, str] = {}

    for elem in root.iter():
        if elem.tag.endswith("Weakness"):
            cwe_id = elem.attrib.get("ID")
            if not cwe_id:
                continue
            name_map[cwe_id] = elem.attrib.get("Name", "")

            for child in elem:
                if child.tag.endswith("Related_Weaknesses"):
                    for rel in child:
                        if (
                            rel.tag.endswith("Related_Weakness")
                            and rel.attrib.get("Nature") == "ChildOf"
                        ):
                            parent_id = rel.attrib.get("CWE_ID")
                            if parent_id:
                                parent_map[cwe_id].append(parent_id)
                                child_map[parent_id].append(cwe_id)

    return dict(parent_map), dict(child_map), name_map


def get_ancestors(cwe_id: str, parent_map: dict, max_depth: int = 20) -> list[str]:
    """Trace ancestry from leaf to root. Returns [parent, grandparent, ...]."""
    ancestors = []
    current = cwe_id
    visited = {current}
    for _ in range(max_depth):
        parents = parent_map.get(current, [])
        if not parents:
            break
        current = parents[0]  # primary parent (Ordinal=Primary preferred)
        if current in visited:
            break
        ancestors.append(current)
        visited.add(current)
    return ancestors


def get_ancestor_at_depth(cwe_id: str, parent_map: dict, depth: int) -> str | None:
    """Get ancestor at specific depth (depth=1 = direct parent, depth=2 = grandparent)."""
    ancestors = get_ancestors(cwe_id, parent_map)
    if len(ancestors) >= depth:
        return ancestors[depth - 1]
    if ancestors:
        return ancestors[-1]  # return highest available
    return cwe_id  # is root itself


def bfs_distance(start: str, end: str, graph: dict) -> int:
    """BFS shortest path distance between two CWE nodes."""
    if start == end:
        return 0
    if start not in graph or end not in graph:
        return 999

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current, dist = queue.popleft()
        if current == end:
            return dist
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return 999


def build_undirected_graph(parent_map: dict, child_map: dict) -> dict:
    """Build undirected adjacency list for BFS distance computation."""
    graph: dict[str, set] = defaultdict(set)
    for child, parents in parent_map.items():
        for parent in parents:
            graph[child].add(parent)
            graph[parent].add(child)
    for parent, children in child_map.items():
        for child in children:
            graph[parent].add(child)
            graph[child].add(parent)
    return {k: list(v) for k, v in graph.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CWE taxonomy tree.")
    parser.add_argument("--xml", default="data/cwe/699.xml", help="Path to CWE XML file")
    parser.add_argument("--depth", type=int, default=2, help="Ancestor depth for grouping (default: 2)")
    parser.add_argument("--matrix", action="store_true", help="Build distance matrix")
    parser.add_argument("--cwes", nargs="+", default=None, help="CWEs for distance matrix (default: all dataset CWEs)")
    parser.add_argument("--out", default=None, help="Output JSON path for distance matrix")
    args = parser.parse_args()

    print(f"Parsing {args.xml}...")
    parent_map, child_map, name_map = parse_xml(args.xml)

    all_ids = set(name_map.keys())
    print(f"Total weaknesses in XML: {len(all_ids)}")

    # Find roots
    roots = [i for i in all_ids if i not in parent_map]
    print(f"\nRoot nodes ({len(roots)}):")
    for r in sorted(roots):
        print(f"  CWE-{r}: {name_map.get(r, '?')}")

    # Check dataset coverage
    covered = [c for c in DATASET_CWES if c in all_ids]
    missing = [c for c in DATASET_CWES if c not in all_ids]
    print(f"\nDataset CWE coverage: {len(covered)}/{len(DATASET_CWES)}")
    if missing:
        print(f"Missing from XML: {missing}")

    # Build tree-based groups at specified depth
    print(f"\n--- Tree-based groups at ancestor depth={args.depth} ---")
    groups: dict[str, list[str]] = defaultdict(list)
    cwe_to_group: dict[str, str] = {}

    for cwe in sorted(covered, key=lambda x: int(x) if x.isdigit() else 0):
        ancestor = get_ancestor_at_depth(cwe, parent_map, args.depth)
        if ancestor:
            ancestor_name = name_map.get(ancestor, ancestor)
            group_key = f"CWE-{ancestor} ({ancestor_name[:40]})"
            groups[group_key].append(cwe)
            cwe_to_group[cwe] = group_key

    for group, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"\n  {group}  [{len(members)} CWEs]")
        for m in sorted(members, key=lambda x: int(x) if x.isdigit() else 0):
            print(f"    CWE-{m}: {name_map.get(m, '?')}")

    # Full ancestry trace for each dataset CWE
    print(f"\n--- Full ancestry paths ---")
    for cwe in sorted(covered, key=lambda x: int(x) if x.isdigit() else 0):
        ancestors = get_ancestors(cwe, parent_map)
        path = " -> ".join([f"CWE-{a}" for a in ancestors])
        print(f"  CWE-{cwe} ({name_map.get(cwe,'?')[:35]}): {path if path else 'ROOT'}")

    # Distance matrix
    if args.matrix:
        target_cwes = args.cwes if args.cwes else [c for c in DATASET_CWES if c in all_ids]
        print(f"\n--- Distance matrix for {len(target_cwes)} CWEs ---")

        graph = build_undirected_graph(parent_map, child_map)

        n = len(target_cwes)
        raw_matrix = [[0] * n for _ in range(n)]
        max_dist = 0

        for i in range(n):
            for j in range(n):
                d = bfs_distance(target_cwes[i], target_cwes[j], graph)
                raw_matrix[i][j] = d
                if d != 999 and d > max_dist:
                    max_dist = d

        # Normalize 0.0 → 1.0
        normalized = []
        for i in range(n):
            row = []
            for j in range(n):
                v = raw_matrix[i][j]
                row.append(round(v / max_dist if v != 999 else 1.0, 3))
            normalized.append(row)

        out_path = args.out or "data/cwe/cwe_distance_matrix.json"
        output = {
            "cwe_order": target_cwes,
            "cwe_names": {c: name_map.get(c, "") for c in target_cwes},
            "max_raw_distance": max_dist,
            "distance_matrix": normalized,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved distance matrix -> {out_path}")

        # Print raw distance table for key CWEs
        print("\nRaw distance sample (first 10 CWEs):")
        header = "      " + "  ".join(f"{c:>5}" for c in target_cwes[:10])
        print(header)
        for i in range(min(10, n)):
            row_str = "  ".join(f"{raw_matrix[i][j]:5}" for j in range(min(10, n)))
            print(f"  {target_cwes[i]:>4}: {row_str}")


if __name__ == "__main__":
    main()
