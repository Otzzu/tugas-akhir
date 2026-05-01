"""
hierarchical_supcon.py — Hierarchical Supervised Contrastive Loss

Positive pair weighting based on CWE group hierarchy:
  - Same CWE class (y_i == y_j, both > 0) → weight 1.0
  - Same group, different CWE (group_i == group_j, y_i != y_j, both > 0) → weight alpha
  - Different group or benign → pure negative (not counted as positive)

Benign samples (y == 0) are excluded from the anchor set but still serve as
negatives in the denominator, pushing vulnerable embeddings away from benign.

Loss for anchor i:
    L_i = -1/|P_i| * sum_{j in P_i} w_ij * log [ exp(z_i·z_j / tau) /
                                                   sum_{k != i} exp(z_i·z_k / tau) ]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalSupConLoss(nn.Module):
    """
    Parameters
    ----------
    temperature : float
        Softmax temperature (default 0.07).
    alpha : float
        Positive weight for same-group, different-CWE pairs (default 0.5).
        Hard positive (same CWE) weight is always 1.0.
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z         : [B, D] embeddings (Z_combined before heads)
        labels    : [B]    fine-grained class IDs (0 = benign)
        group_ids : [B]    coarse group IDs (0 = benign group)

        Returns
        -------
        Scalar contrastive loss.
        """
        device = z.device

        # Only vulnerable samples participate as anchors
        vuln_mask = labels > 0
        n_vuln = int(vuln_mask.sum().item())
        if n_vuln < 2:
            return torch.tensor(0.0, device=device)

        # L2-normalise ALL embeddings (vulns used as anchors, all serve as negatives)
        z_norm = F.normalize(z, dim=-1)  # [B, D]

        z_anc = z_norm[vuln_mask]           # [A, D]  anchors
        l_anc = labels[vuln_mask]           # [A]
        g_anc = group_ids[vuln_mask]        # [A]

        # Similarity: anchors × all samples [A, B]
        sim = torch.matmul(z_anc, z_norm.T) / self.temperature  # [A, B]

        # Positive weight matrix for anchor-to-all pairs [A, B]
        # anchor i vs sample j:
        #   w=1.0 if same CWE (both vuln, y_i==y_j, j != anchor_idx)
        #   w=alpha if same group, diff CWE (both vuln)
        #   w=0.0 otherwise (different group, or j==i, or benign)

        l_all = labels.unsqueeze(0)         # [1, B]
        g_all = group_ids.unsqueeze(0)      # [1, B]

        l_anc_exp = l_anc.unsqueeze(1)      # [A, 1]
        g_anc_exp = g_anc.unsqueeze(1)      # [A, 1]
        vuln_all = (l_all > 0)              # [1, B]

        same_cwe   = (l_anc_exp == l_all) & vuln_all          # [A, B]
        same_group = (g_anc_exp == g_all) & ~same_cwe & vuln_all  # [A, B]

        # Zero out self-pairs (anchor i is also sample j when vuln_mask is used)
        # Build anchor indices in full batch space
        anc_idx = torch.where(vuln_mask)[0]               # [A] — positions in [B]
        self_mask = torch.zeros(n_vuln, z.size(0), dtype=torch.bool, device=device)
        self_mask[torch.arange(n_vuln, device=device), anc_idx] = True

        same_cwe   = same_cwe   & ~self_mask
        same_group = same_group & ~self_mask

        weights = torch.zeros(n_vuln, z.size(0), device=device)
        weights[same_cwe]   = 1.0
        weights[same_group] = self.alpha

        has_pos = (weights > 0).any(dim=1)   # [A]
        if not has_pos.any():
            return torch.tensor(0.0, device=device)

        # Numerical stability: subtract row max before exp
        sim_stable = sim - sim.detach().amax(dim=1, keepdim=True)

        # Denominator: exp-sum over all j != i
        exp_all = torch.exp(sim_stable)                       # [A, B]
        exp_all = exp_all.masked_fill(self_mask, 0.0)         # zero self-pair

        log_denom = torch.log(exp_all.sum(dim=1, keepdim=True).clamp(min=1e-8))  # [A, 1]
        log_prob = sim_stable - log_denom                     # [A, B]

        n_pos = weights.sum(dim=1).clamp(min=1e-8)           # [A]
        per_anchor = -(weights * log_prob).sum(dim=1) / n_pos  # [A]

        return per_anchor[has_pos].mean()
