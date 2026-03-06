"""
dataset.py
~~~~~~~~~~
PyTorch Geometric InMemoryDataset for vulnerability detection.

The dataset scans `data/raw/` for **subdirectories** — one per class.
Subdirectory names become the class labels (sorted alphabetically).

Example for binary:
  data/raw/
    benign/       → label 0
    vulnerable/   → label 1

Example for multi-class (CWE types):
  data/raw/
    benign/       → label 0
    cwe_119/      → label 1
    cwe_120/      → label 2
    cwe_476/      → label 3

Each subdir may contain Joern-exported .json or .graphml CPG files.
The processed PyG dataset is cached to `data/processed/` to avoid re-computation.
Delete `data/processed/vulnerability_dataset.pt` to force re-processing.

Usage
-----
    from gnn_vuln.data.dataset import VulnerabilityDataset
    ds = VulnerabilityDataset(root="data")
    print(ds.class_names)   # ['benign', 'cwe_119', ...]
    print(ds.num_classes)   # 3
    print(ds[0])            # Data(x=[N, 40], edge_index=[2, E], y=[1])
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset

from gnn_vuln.data.graph_builder import build_graph_from_json, build_graph_from_graphml
from gnn_vuln.data.preprocess import preprocess


class VulnerabilityDataset(InMemoryDataset):
    """
    Dataset of CPG graphs for multi-class (or binary) vulnerability classification.

    Class labels are assigned by sorting subdirectory names alphabetically.
    Set `model.num_classes` in `configs/default.yaml` to match the number
    of class subdirectories.

    Parameters
    ----------
    root : str | Path
        Root data directory. Should contain `raw/` subdirectory with per-class subdirs.
    max_nodes : int
        Maximum number of nodes per graph — larger graphs are dropped.
    transform : callable, optional
        PyG transform applied at access time.
    pre_transform : callable, optional
        PyG transform applied at preprocessing time.
    """

    def __init__(
        self,
        root: str | Path = "data",
        max_nodes: int = 500,
        transform=None,
        pre_transform=None,
    ):
        self.max_nodes = max_nodes
        super().__init__(str(root), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        """Return empty — data is provided manually, no automatic download."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ["vulnerability_dataset.pt"]

    def download(self) -> None:
        # Data must be provided manually — see README for Joern instructions
        pass

    def process(self) -> None:
        raw_dir = Path(self.raw_dir)

        # ---------------------------------------------------------------
        # Discover class directories dynamically (sorted → deterministic)
        # ---------------------------------------------------------------
        class_dirs = sorted(
            [d for d in raw_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        if not class_dirs:
            raise RuntimeError(
                f"No class subdirectories found in {raw_dir}. "
                "Create one subdirectory per class (e.g. data/raw/benign/, data/raw/cwe_119/)."
            )

        class_names = [d.name for d in class_dirs]
        class_to_label = {name: idx for idx, name in enumerate(class_names)}

        data_list: list[Data] = []

        for cls_dir in class_dirs:
            label = class_to_label[cls_dir.name]
            for graph_file in cls_dir.iterdir():
                graph: Data | None = None
                if graph_file.suffix == ".json":
                    graph = build_graph_from_json(graph_file, label=label, max_nodes=self.max_nodes)
                elif graph_file.suffix == ".graphml":
                    graph = build_graph_from_graphml(graph_file, label=label, max_nodes=self.max_nodes)
                # (skip .gitkeep and other non-graph files silently)

                if graph is not None:
                    if self.pre_transform is not None:
                        graph = self.pre_transform(graph)
                    data_list.append(graph)

        if not data_list:
            raise RuntimeError(
                "No graphs found in any class subdirectory. "
                "Make sure subdirs contain Joern-exported .json or .graphml files."
            )

        # Save class metadata alongside the graph data
        data, slices = self.collate(data_list)
        torch.save(
            (data, slices, class_names),
            self.processed_paths[0],
        )

    # ------------------------------------------------------------------
    # Override load to also restore class_names
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def _load(self):  # called by __init__ via super().__init__
        pass

    # Patch: re-load to get class_names
    def __init__(  # noqa: F811  (re-define to add class_names restore)
        self,
        root: str | Path = "data",
        max_nodes: int = 500,
        transform=None,
        pre_transform=None,
    ):
        self.max_nodes = max_nodes
        super(VulnerabilityDataset, self).__init__(str(root), transform, pre_transform)
        result = torch.load(self.processed_paths[0])
        if len(result) == 3:
            self.data, self.slices, self.class_names = result
        else:
            self.data, self.slices = result
            self.class_names = None

    @property
    def num_classes(self) -> int:
        if self.class_names:
            return len(self.class_names)
        return int(self.data.y.max().item()) + 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_splits(self, train_ratio=0.7, val_ratio=0.15, seed=42) -> tuple:
        """Return (train_idx, val_idx, test_idx) index lists."""
        import random
        indices = list(range(len(self)))
        random.seed(seed)
        random.shuffle(indices)
        n = len(indices)
        t = int(n * train_ratio)
        v = int(n * val_ratio)
        return indices[:t], indices[t : t + v], indices[t + v :]

