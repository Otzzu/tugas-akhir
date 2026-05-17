"""GNN vulnerability detection models."""

from gnn_vuln.models.base import VulnDetectorBase
from gnn_vuln.models.registry import MODEL_REGISTRY, build_model, _parse_active_heads
from gnn_vuln.models.lmgat_codebert import LMGATCodeBERTVulnDetector
from gnn_vuln.models.lmgat_codebert_mtl import LMGATCodeBERTMTLVulnDetector
from gnn_vuln.models.lmgat_mcs import LMGATMCSVulnDetector
from gnn_vuln.models.lmgat_interp import LMGATInterpVulnDetector
from gnn_vuln.models.lmgat_seq import LMGATSeqVulnDetector
from gnn_vuln.models.lmgat_waves_seq import LMGATWavesSeqVulnDetector
from gnn_vuln.models.lmgat_dualflow import LMGATDualFlowVulnDetector
from gnn_vuln.models.lmgat_hcdfgat import LMGATHCDFGATVulnDetector

__all__ = [
    "VulnDetectorBase",
    "MODEL_REGISTRY",
    "build_model",
    "_parse_active_heads",
    "LMGATCodeBERTVulnDetector",
    "LMGATCodeBERTMTLVulnDetector",
    "LMGATMCSVulnDetector",
    "LMGATInterpVulnDetector",
    "LMGATSeqVulnDetector",
    "LMGATWavesSeqVulnDetector",
    "LMGATDualFlowVulnDetector",
    "LMGATHCDFGATVulnDetector",
]
