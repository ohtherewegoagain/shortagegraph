"""R-GCN encoder with a monotone multi-horizon hazard head (Section III).

Message passing is written directly in PyTorch as in Eq. (3): per-relation
linear maps, mean aggregation over neighbours of that relation, residual-free
sum, ReLU then layer norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .graph import RELATIONS


class RelationalLayer(nn.Module):
    def __init__(self, dim: int, relations: tuple[str, ...] = RELATIONS):
        super().__init__()
        self.relations = relations
        self.self_lin = nn.Linear(dim, dim)
        self.rel_lins = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in relations)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        out = self.self_lin(h)
        num_nodes = h.size(0)
        for r, lin in enumerate(self.rel_lins):
            mask = edge_type == r
            if not bool(mask.any()):
                continue
            src = edge_index[0, mask]
            dst = edge_index[1, mask]
            msg = lin(h[src])
            agg = torch.zeros_like(h).index_add_(0, dst, msg)
            deg = torch.zeros(num_nodes, dtype=h.dtype).index_add_(
                0, dst, torch.ones_like(dst, dtype=h.dtype)
            )
            out = out + agg / deg.clamp(min=1.0).unsqueeze(-1)
        return self.norm(torch.relu(out))


class ShortageGraphNet(nn.Module):
    """Heterogeneous encoder + hazard head producing cumulative onset risk."""

    def __init__(
        self,
        drug_features: int,
        labeler_features: int,
        estab_features: int,
        *,
        dim: int = 32,
        layers: int = 2,
        horizons: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.drug_proj = nn.Linear(drug_features, dim)
        self.labeler_proj = nn.Linear(labeler_features, dim)
        self.estab_proj = nn.Linear(estab_features, dim)
        self.blocks = nn.ModuleList(RelationalLayer(dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, horizons)
        self.horizons = horizons

    def forward(
        self,
        Xd: torch.Tensor,
        Xl: torch.Tensor,
        Xe: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        n_d, n_l, n_e = Xd.size(0), Xl.size(0), Xe.size(0)
        h = torch.zeros(n_d + n_l + n_e, self.drug_proj.out_features, dtype=Xd.dtype)
        if n_d:
            h[:n_d] = self.drug_proj(Xd)
        if n_l:
            h[n_d : n_d + n_l] = self.labeler_proj(Xl)
        if n_e:
            h[n_d + n_l :] = self.estab_proj(Xe)
        for block in self.blocks:
            h = self.dropout(block(h, edge_index, edge_type))
        drug_repr = h[:n_d]
        hazard = torch.sigmoid(self.head(drug_repr))          # (n_d, H)
        # cumulative onset probability: 1 - prod(1 - lambda)
        survival = torch.cumprod(1.0 - hazard, dim=1)
        return 1.0 - survival


class EdgeFreeMLP(nn.Module):
    """Same drug features, no message passing: isolates the value of structure."""

    def __init__(self, drug_features: int, *, hidden: int = 64, horizons: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(drug_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizons),
        )

    def forward(self, Xd: torch.Tensor, *args) -> torch.Tensor:
        hazard = torch.sigmoid(self.net(Xd))
        return 1.0 - torch.cumprod(1.0 - hazard, dim=1)


def weighted_bce(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, alpha: float) -> torch.Tensor:
    """Class-weighted BCE summed over horizons (Eq. 6), at-risk drugs only."""
    if not bool(mask.any()):
        return pred.sum() * 0.0
    pred = pred[mask]
    target = target[mask]
    eps = 1e-7
    loss = -(
        alpha * target * torch.log(pred.clamp(min=eps))
        + (1.0 - target) * torch.log((1.0 - pred).clamp(min=eps))
    )
    return loss.sum()


def positive_weight(y) -> float:
    """Square root of the negative-to-positive ratio, as in Section III-C."""
    import numpy as np

    arr = np.asarray(y).ravel()
    pos = float(arr.sum())
    neg = float(arr.size - pos)
    if pos <= 0:
        return 1.0
    return float(np.sqrt(neg / pos))
