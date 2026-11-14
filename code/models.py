"""Models: Tab-Transformer (paper Sec. 6.2) and MLP baseline, plus Focal Loss.

Tab-Transformer per the paper:
  - each scalar feature x_j -> learned linear projection e_j in R^d  (Eq. 2)
  - L transformer blocks (MHSA + FF + LayerNorm, residual)          (Eq. 3)
  - global average pooling + MLP head -> risk score in [0,1]        (Eq. 4)

Opacus compatibility: uses LayerNorm (not BatchNorm) throughout, so the
model passes Opacus' module validator unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.layers import DPMultiheadAttention


class FeatureTokenizer(nn.Module):
    """Per-feature learned linear projection: e_j = W_j x_j + b_j.

    Implemented as a grouped 1x1 Conv1d (groups = n_features) so that each
    feature has its own independent (d)-dim projection AND Opacus can compute
    per-sample gradients (Conv1d is a natively supported layer).
    """

    def __init__(self, n_features: int, d_embed: int):
        super().__init__()
        self.n_features, self.d_embed = n_features, d_embed
        self.proj = nn.Conv1d(n_features, n_features * d_embed,
                              kernel_size=1, groups=n_features)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):                      # x: (B, F)
        h = self.proj(x.unsqueeze(-1))         # (B, F*d, 1)
        return h.squeeze(-1).view(-1, self.n_features, self.d_embed)


class TransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        # DPMultiheadAttention: Opacus' drop-in replacement supporting
        # per-example gradients; batch_first=True keeps (B, F, d) layout so
        # the grad-sample hooks see the correct batch dimension.
        self.attn = DPMultiheadAttention(d, heads, dropout=dropout,
                                         batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff_dim), nn.ReLU(),
                                nn.Linear(ff_dim, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # x: (B, F, d)
        h, _ = self.attn(x, x, x)
        x = self.ln1(x + self.drop(h))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class TabTransformer(nn.Module):
    def __init__(self, n_features: int, d_embed=32, n_heads=4, n_blocks=2,
                 ff_dim=64, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_embed)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_embed, n_heads, ff_dim, dropout)
             for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Linear(d_embed, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        h = self.tokenizer(x)
        for blk in self.blocks:
            h = blk(h)
        pooled = h.mean(dim=1)                 # global average pool over tokens
        return self.head(pooled).squeeze(-1)   # logits


class MLPBaseline(nn.Module):
    def __init__(self, n_features: int, hidden=(64, 32)):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def focal_loss(logits, targets, alpha=0.75, gamma=2.0, pos_weight=None,
               reduction="mean"):
    """Binary focal loss (paper Eq. 5) on logits.

    pos_weight: optional extra weight on the positive class — the DP-safe
    replacement for record oversampling (no data duplication).
    """
    p = torch.sigmoid(logits)
    t = targets.float()
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p_t = p * t + (1 - p) * (1 - t)
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    loss = alpha_t * (1 - p_t).clamp(min=1e-8) ** gamma * ce
    if pos_weight is not None:
        loss = loss * (1.0 + (pos_weight - 1.0) * t)
    if reduction == "mean":
        return loss.mean()
    return loss


def build_model(name: str, n_features: int, cfg) -> nn.Module:
    if name == "tabtransformer":
        return TabTransformer(n_features, cfg.d_embed, cfg.n_heads,
                              cfg.n_blocks, cfg.ff_dim,
                              getattr(cfg, 'dropout', 0.1))
    if name == "mlp":
        return MLPBaseline(n_features, cfg.mlp_hidden)
    raise ValueError(name)
