"""
gnn_vuln — GNN-based Vulnerability Detection
=============================================
Final project package.
"""

from importlib.metadata import PackageNotFoundError, version as _version

# read from the installed metadata, never a second literal: this string is written into every
# dataset fingerprint, and a hand-bumped copy drifts (it sat at rc3 through the rc4 release).
try:
    __version__ = _version("gnn-vuln")
except PackageNotFoundError:      # not installed (running straight from src/)
    __version__ = "0.0.0.dev0"

__author__ = "Otzzu"
