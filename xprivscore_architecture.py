"""
X-PrivScore Enhanced Architecture
==================================
Replaces plain MLP with a hybrid Tab-Transformer + MLP architecture.
- Tab-Transformer: self-attention over categorical/tabular features
- MLP head: final risk scoring
- Federated-ready: same API as original MLP, drop-in replacement
- DP-compatible: gradient clipping works unchanged

Usage:
    model = XPrivScoreTransformer(input_dim=13)
    # identical to original MLP usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
# 1. Multi-Head Self-Attention Block
# ─────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention for tabular features.
    Each feature is treated as a token.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, embed_dim)"""
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)          # each (B, N, heads, head_dim)
        q = q.transpose(1, 2)                 # (B, heads, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale   # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


# ─────────────────────────────────────────────
# 2. Transformer Encoder Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn   = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff     = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1  = nn.LayerNorm(embed_dim)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ─────────────────────────────────────────────
# 3. Feature Embedding Layer
# ─────────────────────────────────────────────
class FeatureEmbedding(nn.Module):
    """
    Projects each scalar feature into a shared embedding space.
    Treats every feature as a token (Tab-Transformer style).
    """
    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        # One linear projection per feature: scalar -> embed_dim
        self.embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_features)
        ])
        self.num_features = num_features
        self.embed_dim    = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_features)
        returns: (batch, num_features, embed_dim)
        """
        tokens = [
            emb(x[:, i].unsqueeze(-1))          # (B, embed_dim)
            for i, emb in enumerate(self.embeddings)
        ]
        return torch.stack(tokens, dim=1)        # (B, num_features, embed_dim)


# ─────────────────────────────────────────────
# 4. Main Hybrid Model: Tab-Transformer + MLP
# ─────────────────────────────────────────────
class XPrivScoreTransformer(nn.Module):
    """
    Drop-in replacement for the original MLP in X-PrivScore.

    Architecture:
        Input (batch, input_dim)
            ↓
        FeatureEmbedding  → (batch, input_dim, embed_dim)
            ↓
        N × TransformerBlock
            ↓
        Global Average Pool → (batch, embed_dim)
            ↓
        MLP Head (64 → 32 → 1)
            ↓
        Sigmoid → risk score ∈ [0, 1]

    Args:
        input_dim   : number of input features (default 13)
        embed_dim   : transformer token dimension (default 32)
        num_heads   : attention heads (default 4)
        num_layers  : transformer depth (default 2)
        ff_dim      : feedforward hidden dim (default 64)
        dropout     : dropout rate (default 0.1)
    """
    def __init__(
        self,
        input_dim:  int = 13,
        embed_dim:  int = 32,
        num_heads:  int = 4,
        num_layers: int = 2,
        ff_dim:     int = 64,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.embedding = FeatureEmbedding(input_dim, embed_dim)

        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # MLP classification head (matches original architecture)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim)
        returns: (batch,) raw logits  ← sigmoid applied in loss/evaluate
        """
        tokens = self.embedding(x)              # (B, F, E)
        tokens = self.transformer(tokens)       # (B, F, E)
        pooled = tokens.mean(dim=1)             # (B, E)
        logit  = self.head(pooled).squeeze(-1)  # (B,)
        return logit   # raw logit, no sigmoid here

    def get_attention_weights(self, x: torch.Tensor):
        """
        Returns attention weights from the first transformer block.
        Useful for SHAP-style feature importance analysis.
        """
        tokens = self.embedding(x)
        block  = self.transformer[0]
        with torch.no_grad():
            normed = block.norm1(tokens)
            B, N, D = normed.shape
            qkv = block.attn.qkv(normed).reshape(
                B, N, 3, block.attn.num_heads, block.attn.head_dim)
            q, k, _ = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            attn = F.softmax(
                (q @ k.transpose(-2, -1)) / block.attn.scale, dim=-1
            )
        return attn.mean(dim=1)   # (B, N, N) averaged over heads


# ─────────────────────────────────────────────
# 5. Original MLP (kept for ablation comparison)
# ─────────────────────────────────────────────
class XPrivScoreMLP(nn.Module):
    """Original baseline MLP — keep for ablation study."""
    def __init__(self, input_dim: int = 13, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # raw logit


# ─────────────────────────────────────────────
# 6. Federated Training Helper
# ─────────────────────────────────────────────
def get_model_update(global_model: nn.Module,
                     local_model:  nn.Module) -> dict:
    """Compute weight delta for FedAvg upload."""
    delta = {}
    g = dict(global_model.named_parameters())
    l = dict(local_model.named_parameters())
    for name in g:
        delta[name] = (l[name].data - g[name].data).clone()
    return delta


def apply_fedavg(global_model: nn.Module,
                 deltas: list,
                 weights: list) -> nn.Module:
    """
    Weighted FedAvg aggregation.
    deltas : list of weight-delta dicts from each client
    weights: list of dataset sizes (for proportional averaging)
    """
    total = sum(weights)
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            agg = sum(
                (w / total) * d[name]
                for w, d in zip(weights, deltas)
            )
            param.data += agg
    return global_model


# ─────────────────────────────────────────────
# 7. DP Noise Injection (Gaussian Mechanism)
# ─────────────────────────────────────────────
def clip_and_noise(model: nn.Module,
                   clip_norm: float = 1.0,
                   noise_sigma: float = 1.2) -> nn.Module:
    """
    Apply gradient clipping + Gaussian DP noise before upload.
    Call after loss.backward() and before optimizer.step().
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = min(1.0, clip_norm / (total_norm + 1e-6))
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(clip_coef)
            p.grad.data.add_(
                torch.randn_like(p.grad.data) * noise_sigma * clip_norm
            )
    return model


# ─────────────────────────────────────────────
# 8. Quick Sanity Check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    batch = torch.randn(32, 13)   # 32 transactions, 13 features

    # --- Transformer model ---
    model_t = XPrivScoreTransformer(input_dim=13)
    out_t   = model_t(batch)
    print(f"Transformer output shape : {out_t.shape}")       # (32,)
    print(f"Transformer param count  : "
          f"{sum(p.numel() for p in model_t.parameters()):,}")

    # --- Baseline MLP ---
    model_m = XPrivScoreMLP(input_dim=13)
    out_m   = model_m(batch)
    print(f"MLP output shape         : {out_m.shape}")       # (32,)
    print(f"MLP param count          : "
          f"{sum(p.numel() for p in model_m.parameters()):,}")

    # --- Attention weights (for explainability) ---
    attn = model_t.get_attention_weights(batch)
    print(f"Attention weights shape  : {attn.shape}")        # (32, 13, 13)

    print("\n✅ Architecture check passed.")
