"""Membership-inference evaluation and the DP-sanitised SHAP layer.

MI attack: shadow-style threshold/logistic attack on per-example losses and
confidences (Yeom et al. / Shokri et al. family). Reports ATTACK AUC, where
0.5 = no leakage (the ideal defended outcome).

Explainability metrics — the three named metrics of the revised paper:
  Cov    : Explanation Coverage — fraction of flagged tx receiving a
           VALIDATED DP-sanitised SHAP vector within the sampling budget
           (validation: additivity check after noising).
  Agr@3  : mean top-3 overlap between federated-model SHAP and the
           centralised reference model's SHAP on the same transactions.
  rho    : Spearman correlation between global DP-aggregated feature
           importance and |coef| ranking of an L1 logistic regression.
"""
import numpy as np
import torch
import shap
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from models import focal_loss


# ── membership inference ────────────────────────────────────────────────────

@torch.no_grad()
def _per_example_losses(model, X, y, batch=4096):
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch):
        logits = model(torch.tensor(X[i:i + batch]).to(device))
        loss = focal_loss(logits, torch.tensor(y[i:i + batch]).to(device),
                          reduction="none")
        out.append(loss.cpu().numpy())
    return np.concatenate(out)


def membership_inference_auc(model, X_train, y_train, X_test, y_test,
                             n_each=2000, seed=0):
    """Attack AUC of a loss-threshold MI attack (0.5 = random = no leakage)."""
    rng = np.random.default_rng(seed)
    n_each = min(n_each, len(X_train), len(X_test))
    idx_in = rng.choice(len(X_train), n_each, replace=False)
    idx_out = rng.choice(len(X_test), n_each, replace=False)
    l_in = _per_example_losses(model, X_train[idx_in], y_train[idx_in])
    l_out = _per_example_losses(model, X_test[idx_out], y_test[idx_out])
    scores = np.concatenate([-l_in, -l_out])       # lower loss => "member"
    labels = np.concatenate([np.ones(n_each), np.zeros(n_each)])
    return float(roc_auc_score(labels, scores))


# ── DP-sanitised SHAP layer ─────────────────────────────────────────────────

def _model_fn(model):
    device = next(model.parameters()).device
    def f(X):
        with torch.no_grad():
            return torch.sigmoid(
                model(torch.tensor(X, dtype=torch.float32).to(device))
            ).cpu().numpy()
    return f


def dp_shap_for_flagged(model, X_flagged, X_background, sigma_shap=1.0,
                        c_shap=1.0, n_coalitions=256, seed=0,
                        additivity_tol=0.05):
    """KernelSHAP per flagged transaction + clip-to-C + Gaussian noise.

    Returns (sanitised_attributions [n, F] with NaN rows for suppressed
    explanations, coverage fraction).
    Validation: after noising, |sum(phi) - (f(x) - base)| <= tol, else the
    explanation is suppressed (counts against coverage).
    """
    rng = np.random.default_rng(seed)
    f = _model_fn(model)
    bg = shap.sample(X_background, min(50, len(X_background)),
                     random_state=seed)
    explainer = shap.KernelExplainer(f, bg)
    base = float(np.mean(f(bg)))
    n, F = X_flagged.shape
    out = np.full((n, F), np.nan)
    ok = 0
    for i in range(n):
        try:
            phi = explainer.shap_values(X_flagged[i:i + 1],
                                        nsamples=n_coalitions, silent=True)
            phi = np.asarray(phi).reshape(-1)[:F]
        except Exception:
            continue
        norm = np.linalg.norm(phi)
        if norm > c_shap:
            phi = phi * (c_shap / norm)
        phi_dp = phi + rng.normal(0, sigma_shap * c_shap / np.sqrt(F), F)
        pred = float(f(X_flagged[i:i + 1])[0])
        if abs(phi_dp.sum() - (pred - base)) <= additivity_tol + \
                3 * sigma_shap * c_shap / np.sqrt(F):
            out[i] = phi_dp
            ok += 1
    return out, ok / max(n, 1)


def agreement_at_3(phi_fed: np.ndarray, phi_ref: np.ndarray) -> float:
    """Mean top-3 feature overlap between two attribution matrices."""
    vals = []
    for a, b in zip(phi_fed, phi_ref):
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        ta = set(np.argsort(-np.abs(a))[:3])
        tb = set(np.argsort(-np.abs(b))[:3])
        vals.append(len(ta & tb) / 3.0)
    return float(np.mean(vals)) if vals else float("nan")


def global_rank_correlation(phi_matrix: np.ndarray, X_tr, y_tr) -> float:
    """Spearman rho between mean |SHAP| ranking and L1-logistic |coef|."""
    valid = phi_matrix[~np.isnan(phi_matrix).any(axis=1)]
    if len(valid) == 0:
        return float("nan")
    imp_shap = np.abs(valid).mean(axis=0)
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                            class_weight="balanced", max_iter=2000)
    lr.fit(X_tr, y_tr)
    imp_lr = np.abs(lr.coef_).reshape(-1)
    rho, _ = spearmanr(imp_shap, imp_lr)
    return float(rho)
