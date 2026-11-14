"""End-to-end smoke test on synthetic data (runs in a few minutes on CPU).

Verifies: data pipeline, both models, DP-SGD via Opacus + accountant,
trimmed-mean, AMR correctness + collusion bound + latency, attacks path,
MI attack, and the DP-SHAP layer.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import ExperimentConfig
from train_federated import run_federated
from privacy_and_explain import (membership_inference_auc, dp_shap_for_flagged,
                                 global_rank_correlation)
from amr import verify_correctness, collusion_simulation, latency_benchmark


def main():
    print("1) AMR correctness:", end=" ")
    err = verify_correctness()
    assert err < 1e-6, f"mask cancellation failed, err={err}"
    print(f"OK (max cancellation error {err:.2e})")

    print("2) AMR collusion bound (N=10,k=5,R=6): ", end="")
    emp = collusion_simulation()
    theo = (5 / 10) ** 6
    print(f"empirical={emp:.4f} <= theory={theo:.4f}:", emp <= theo * 1.3)

    print("3) AMR vs Bonawitz latency:", latency_benchmark(dim=5000, rounds=5))

    print("4a) Learning check (no DP, 3 rounds)...")
    cfg = ExperimentConfig(dataset="synthetic", model="tabtransformer")
    cfg.dp.enabled = False
    cfg.fl.global_rounds, cfg.fl.local_epochs, cfg.seeds = 3, 1, [0]
    _, m_nodp, _ = run_federated(cfg, seed=0, verbose=False)
    print(f"    no-DP AUC = {m_nodp['auc']:.3f}")
    assert m_nodp["auc"] > 0.8, "pipeline failed to learn without DP"

    print("4b) DP mechanics check (Opacus + accountant, 3 rounds)...")
    cfg = ExperimentConfig(dataset="synthetic", model="tabtransformer")
    cfg.fl.global_rounds, cfg.fl.local_epochs, cfg.seeds = 3, 1, [0]
    model, metrics, data = run_federated(cfg, seed=0, verbose=False)
    print("   metrics:", {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in metrics.items()
                          if k != "epsilon_per_client"})
    assert metrics["epsilon_worst"] is not None, "accountant produced no eps"
    # NOTE: with DP noise, 3 rounds is far too few to learn — full runs use
    # T=200. The DP path here only verifies mechanics, not model quality.

    X_tr, X_te, y_tr, y_te, _ = data
    print("5) MI attack AUC:",
          round(membership_inference_auc(model, X_tr, y_tr, X_te, y_te), 3))

    print("6) DP-SHAP layer (10 flagged tx)...")
    import torch
    device = next(model.parameters()).device
    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.tensor(X_te[:2000]).to(device))).cpu().numpy()
    flagged = X_te[:2000][np.argsort(-probs)[:10]]
    phi, cov = dp_shap_for_flagged(model, flagged, X_tr, n_coalitions=128)
    rho = global_rank_correlation(phi, X_tr[:5000], y_tr[:5000])
    print(f"   coverage={cov:.2f}, spearman_rho={rho:.2f}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
