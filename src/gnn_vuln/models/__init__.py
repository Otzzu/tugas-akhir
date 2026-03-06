"""GNN models subpackage."""

from gnn_vuln.models.gcn import GCNVulnDetector
from gnn_vuln.models.gat import GATVulnDetector

__all__ = ["GCNVulnDetector", "GATVulnDetector"]
