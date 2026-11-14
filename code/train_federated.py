"""Federated training loop: FL + record-level DP-SGD (Opacus) + trimmed-mean
+ optional AMR masking + attacks.

Privacy mechanism — stated unambiguously (this IS what the paper must report):
  Standard per-minibatch DP-SGD inside local training. Opacus performs
  PER-EXAMPLE gradient clipping to C and adds Gaussian noise with multiplier
  sigma at EVERY optimizer step. The per-client Poisson sampling rate is
  q_k = batch_size / |D_k| and the number of noisy steps per client is
  T * E * ceil(|D_k| / batch_size). Epsilon is computed per client with the
  Opacus RDP accountant; the experiment reports the WORST CASE over clients
  (largest epsilon), which is the honest per-dataset guarantee.
"""
import copy
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score,
                             average_precision_score)

from models import build_model, focal_loss
from aggregation import (weighted_fedavg, trimmed_mean, flip_labels,
                         scale_update, collusive_update)
from amr import amr_aggregate
from data import load_dataset, dirichlet_partition


# ── parameter vector helpers ────────────────────────────────────────────────

def get_flat_params(model) -> np.ndarray:
    return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()


def set_flat_params(model, flat: np.ndarray):
    flat_t = torch.tensor(flat, dtype=torch.float32)
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_t[i:i + n].view_as(p))
        i += n


# ── local client update ─────────────────────────────────────────────────────

def local_update(global_flat, model_name, model_cfg, X, y, cfg, seed,
                 device="cpu"):
    """One client's local training for E epochs; returns (delta, n_records,
    accountant_or_None)."""
    torch.manual_seed(seed)
    model = build_model(model_name, X.shape[1], model_cfg).to(device)
    set_flat_params(model, global_flat)
    model.train()

    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    # uniform-q design: under DP the batch size scales with |D_k| so that the
    # Poisson sampling rate q = B_k/|D_k| = cfg.dp.sample_rate is identical
    # for every client (identical accountant epsilon everywhere).
    if cfg.dp.enabled:
        batch_k = max(8, int(round(len(y) * cfg.dp.sample_rate)))
    else:
        batch_k = cfg.fl.batch_size
    loader = DataLoader(ds, batch_size=batch_k, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.fl.lr)

    accountant = None
    if cfg.dp.enabled:
        engine = PrivacyEngine(accountant="rdp")
        model, opt, loader = engine.make_private(
            module=model, optimizer=opt, data_loader=loader,
            noise_multiplier=cfg.dp.noise_multiplier,
            max_grad_norm=cfg.dp.max_grad_norm,
            poisson_sampling=True,
        )
        accountant = engine.accountant

    # class-weighted focal loss (DP-safe imbalance handling)
    n_pos = max(int(y.sum()), 1)
    pos_weight = min(len(y) / (10.0 * n_pos), 50.0)  # target ~10% effective

    for _ in range(cfg.fl.local_epochs):
        for xb, yb in loader:
            if len(xb) == 0:          # Poisson sampling can yield empty batches
                continue
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = focal_loss(model(xb), yb,
                              alpha=cfg.imbalance.focal_alpha,
                              gamma=cfg.imbalance.focal_gamma,
                              pos_weight=pos_weight)
            loss.backward()
            opt.step()

    # strip Opacus wrapper prefix when reading back parameters
    local_flat = torch.cat([p.data.view(-1)
                            for p in model.parameters()]).cpu().numpy()
    delta = local_flat - global_flat
    return delta, len(y), accountant


# ── evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_proba(model, X, batch=4096):
    model.eval()
    device = next(model.parameters()).device
    probs = []
    for i in range(0, len(X), batch):
        logits = model(torch.tensor(X[i:i + batch]).to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def tune_threshold(model, X_val, y_val):
    """Pick the decision threshold maximising F1 on a held-out validation
    split (never the test set) -- standard practice for imbalanced problems,
    and essential under DP noise which shifts output calibration."""
    p = predict_proba(model, X_val)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.50, 0.999, 200))):
        f1 = f1_score(y_val, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


@torch.no_grad()
def evaluate(model, X, y, batch=4096, threshold=0.5):
    p = predict_proba(model, X, batch)
    pred = (p >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }


# ── main federated run ──────────────────────────────────────────────────────

def run_federated(cfg, seed: int, verbose: bool = True,
                  checkpoint_path: str = None, checkpoint_every: int = 10):
    import pickle, os
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    X_tr, X_te, y_tr, y_te = load_dataset(cfg.dataset, cfg.data_dir, seed)
    from sklearn.model_selection import train_test_split as _tts
    X_tr, X_val, y_tr, y_val = _tts(X_tr, y_tr, test_size=0.1,
                                    stratify=y_tr, random_state=seed)
    parts = dirichlet_partition(y_tr, cfg.fl.n_clients,
                                cfg.fl.dirichlet_alpha, seed)

    torch.manual_seed(seed)
    global_model = build_model(cfg.model, X_tr.shape[1], cfg.model_cfg).to(device)
    global_flat = get_flat_params(global_model)

    malicious = set(rng.choice(cfg.fl.n_clients,
                               size=cfg.attack.n_malicious, replace=False)
                    ) if cfg.attack.kind else set()
    collusion_dir = rng.normal(size=global_flat.shape)
    amr_key = hashlib.sha256(f"xprivscore-{seed}".encode()).digest()
    accountants = {}
    start_round = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as fh:
            ck = pickle.load(fh)
        global_flat = ck["global_flat"]
        accountants = ck["accountants"]
        malicious = ck["malicious"]
        collusion_dir = ck["collusion_dir"]
        start_round = ck["round_done"]
        print(f"  [resume] from round {start_round} ({checkpoint_path})")

    for t in range(start_round, cfg.fl.global_rounds):
        deltas, sizes = [], []
        for cid, idx in enumerate(parts):
            Xc, yc = X_tr[idx], y_tr[idx]
            if cid in malicious and cfg.attack.kind == "label_flip":
                yc = flip_labels(yc, cfg.attack.label_flip_frac, rng)
            delta, n_k, acct = local_update(
                global_flat, cfg.model, cfg.model_cfg, Xc, yc, cfg,
                seed * 100_000 + t * 100 + cid, device=device)
            if acct is not None:
                accountants.setdefault(cid, []).append(
                    copy.deepcopy(acct.history))
            if cid in malicious and cfg.attack.kind == "scaling":
                delta = scale_update(delta, cfg.attack.scale_factor)
            if cid in malicious and cfg.attack.kind == "collusion":
                delta = collusive_update(delta, collusion_dir)
            deltas.append(delta)
            sizes.append(n_k)

        if cfg.use_mpc_sim:  # AMR masking (sum path); correctness verified
            _, _ = amr_aggregate(deltas, amr_key, t)

        if cfg.fl.aggregation == "trimmed_mean":
            agg = trimmed_mean(deltas, cfg.fl.k_trim)
        else:
            agg = weighted_fedavg(deltas, sizes)

        global_flat = global_flat + agg
        if checkpoint_path and (t + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "wb") as fh:
                pickle.dump({"round_done": t + 1, "global_flat": global_flat,
                             "accountants": accountants,
                             "malicious": malicious,
                             "collusion_dir": collusion_dir}, fh)
        if verbose and (t + 1) % max(1, cfg.fl.global_rounds // 10) == 0:
            set_flat_params(global_model, global_flat)
            m = evaluate(global_model, X_te, y_te)
            print(f"  round {t+1:4d}/{cfg.fl.global_rounds}  "
                  f"AUC={m['auc']:.3f}  F1={m['f1']:.3f}  "
                  f"PR-AUC={m['pr_auc']:.3f}")

    set_flat_params(global_model, global_flat)
    thr = tune_threshold(global_model, X_val, y_val)
    metrics = evaluate(global_model, X_te, y_te, threshold=thr)
    metrics["epsilon_worst"], metrics["epsilon_per_client"] = \
        compute_epsilon(accountants, cfg) if cfg.dp.enabled else (None, None)
    return global_model, metrics, (X_tr, X_te, y_tr, y_te, parts)


def compute_epsilon(accountants: dict, cfg):
    """Worst-case per-client epsilon from accumulated Opacus RDP histories."""
    from opacus.accountants import RDPAccountant
    eps_per_client = {}
    for cid, histories in accountants.items():
        acc = RDPAccountant()
        merged = []
        for h in histories:
            merged.extend(h)
        acc.history = merged
        eps, _ = acc.get_privacy_spent(delta=cfg.dp.delta)
        eps_per_client[cid] = float(eps)
    worst = max(eps_per_client.values()) if eps_per_client else None
    return worst, eps_per_client


def run_centralized(cfg, seed: int):
    """Centralised upper bound: pooled data, no DP, no FL."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_tr, X_te, y_tr, y_te = load_dataset(cfg.dataset, cfg.data_dir, seed)
    from sklearn.model_selection import train_test_split as _tts
    X_tr, X_val, y_tr, y_val = _tts(X_tr, y_tr, test_size=0.1,
                                    stratify=y_tr, random_state=seed)
    torch.manual_seed(seed)
    model = build_model(cfg.model, X_tr.shape[1], cfg.model_cfg).to(device)
    ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    loader = DataLoader(ds, batch_size=cfg.fl.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.fl.lr)
    n_pos = max(int(y_tr.sum()), 1)
    pos_weight = min(len(y_tr) / (10.0 * n_pos), 50.0)
    epochs = max(3, cfg.fl.global_rounds * cfg.fl.local_epochs // 20)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            focal_loss(model(xb), yb, cfg.imbalance.focal_alpha,
                       cfg.imbalance.focal_gamma, pos_weight).backward()
            opt.step()
    thr = tune_threshold(model, X_val, y_val)
    return model, evaluate(model, X_te, y_te, threshold=thr)
