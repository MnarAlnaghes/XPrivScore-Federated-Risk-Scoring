"""Central configuration for X-PrivScore experiments.

Every hyperparameter reported in the paper lives here so that the
paper's Section 6.6 / 7.1 can be written directly from this file.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    d_embed: int = 32          # embedding dimension d
    n_heads: int = 4           # attention heads H
    n_blocks: int = 2          # transformer blocks L
    ff_dim: int = 64           # feed-forward dimension
    dropout: float = 0.1       # set 0.0 under DP (standard practice)
    mlp_hidden: tuple = (64, 32)  # MLP baseline (64-32 units)


@dataclass
class DPConfig:
    enabled: bool = True
    noise_multiplier: float = 4.5   # sigma
    max_grad_norm: float = 1.0      # clipping norm C (per-example, Opacus)
    delta: float = 1e-5
    # UNIFORM-q DESIGN: every client uses Poisson sampling at the SAME rate
    # q = sample_rate, achieved by a per-client batch size B_k = q * |D_k|.
    # This makes the number of noisy steps (T*E*round(1/q)) and hence the
    # accountant's epsilon IDENTICAL for every client and every dataset --
    # one clean record-level guarantee for the whole paper.
    sample_rate: float = 0.05       # q = 1/20


@dataclass
class FLConfig:
    n_clients: int = 10             # K
    dirichlet_alpha: float = 0.5    # non-IID concentration
    global_rounds: int = 200        # T
    local_epochs: int = 2           # E
    batch_size: int = 128
    lr: float = 1e-3
    k_trim: int = 1                 # trimmed-mean parameter
    aggregation: str = "trimmed_mean"  # or "fedavg"


@dataclass
class ImbalanceConfig:
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    # DP-safe imbalance handling: class-weighted focal loss (no record
    # duplication). Record oversampling duplicates records and therefore
    # BREAKS strict record-level DP (a duplicated record's sensitivity
    # multiplies). Keep oversample_frac=None when DP is enabled.
    oversample_frac: Optional[float] = None  # e.g. 0.10 (non-DP runs only)


@dataclass
class AttackConfig:
    kind: Optional[str] = None      # None | 'label_flip' | 'scaling' | 'collusion'
    n_malicious: int = 2
    label_flip_frac: float = 0.30   # fraction of local labels flipped
    scale_factor: float = -3.0      # gradient scaling factor (sign-flip amplification)


@dataclass
class ExperimentConfig:
    dataset: str = "ulb"            # 'ulb' | 'ieee' | 'elliptic' | 'synthetic'
    model: str = "tabtransformer"   # 'tabtransformer' | 'mlp'
    use_mpc_sim: bool = True        # simulate AMR masking layer
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    data_dir: str = "data"
    results_dir: str = "results"
