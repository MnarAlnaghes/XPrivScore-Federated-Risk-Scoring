"""Experiment runner.

Examples
--------
Smoke test (synthetic, minutes):
    python run_experiment.py --dataset synthetic --preset smoke

Main result, one dataset, full config (Table 3 row):
    python run_experiment.py --dataset ulb --model tabtransformer
    python run_experiment.py --dataset ulb --model mlp

Baselines (Table 8): --no-dp / --no-mpc / --aggregation fedavg
Attacks   (Table 5): --attack label_flip|scaling|collusion
Privacy sweep (Table 4): --sigma 4.0 / 2.5 / 1.2
Privacy planning only (no training):
    python run_experiment.py --plan --dataset ulb --sigma 1.2

Outputs: results/<name>.json  (mean±std over seeds + per-seed detail +
worst-case epsilon + MI attack AUC + explainability metrics).
"""
import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from config import ExperimentConfig
from train_federated import run_federated, run_centralized, evaluate
from privacy_and_explain import (membership_inference_auc, dp_shap_for_flagged,
                                 agreement_at_3, global_rank_correlation)
from data import load_dataset, dirichlet_partition


def plan_epsilon(sigma, T, E, q, delta):
    """Uniform-q design: epsilon is identical for every client/dataset."""
    from opacus.accountants import RDPAccountant
    steps = T * E * int(round(1.0 / q))
    acc = RDPAccountant()
    acc.history = [(sigma, q, steps)]
    eps, _ = acc.get_privacy_spent(delta=delta)
    return steps, float(eps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--model", default="tabtransformer",
                   choices=["tabtransformer", "mlp"])
    p.add_argument("--preset", default="full", choices=["full", "smoke"])
    p.add_argument("--sigma", type=float, default=4.5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--sample-rate", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-dp", action="store_true")
    p.add_argument("--no-mpc", action="store_true")
    p.add_argument("--aggregation", default="trimmed_mean",
                   choices=["trimmed_mean", "fedavg"])
    p.add_argument("--attack", default=None,
                   choices=[None, "label_flip", "scaling", "collusion"])
    p.add_argument("--n-malicious", type=int, default=2)
    p.add_argument("--n-clients", type=int, default=10)
    p.add_argument("--k-trim", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--centralized", action="store_true")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--plan", action="store_true",
                   help="print per-client epsilon for this config and exit")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    cfg = ExperimentConfig(dataset=args.dataset, model=args.model)
    cfg.dp.noise_multiplier = args.sigma
    cfg.dp.sample_rate = args.sample_rate
    cfg.model_cfg.dropout = args.dropout
    cfg.fl.lr = args.lr
    cfg.fl.local_epochs = args.epochs
    cfg.dp.enabled = not args.no_dp
    cfg.use_mpc_sim = not args.no_mpc
    cfg.fl.aggregation = args.aggregation
    cfg.fl.n_clients = args.n_clients
    cfg.fl.k_trim = args.k_trim
    cfg.attack.kind = args.attack
    cfg.attack.n_malicious = args.n_malicious
    cfg.seeds = args.seeds
    if args.preset == "smoke":
        cfg.fl.global_rounds, cfg.fl.local_epochs, cfg.seeds = 3, 1, [0]

    if args.plan:
        steps, eps = plan_epsilon(cfg.dp.noise_multiplier,
                                  cfg.fl.global_rounds, cfg.fl.local_epochs,
                                  cfg.dp.sample_rate, cfg.dp.delta)
        print(f"uniform-q design: q={cfg.dp.sample_rate}, "
              f"T={cfg.fl.global_rounds}, E={cfg.fl.local_epochs}, "
              f"sigma={cfg.dp.noise_multiplier}, steps={steps}")
        print(f"epsilon = {eps:.2f} at delta={cfg.dp.delta} "
              f"(IDENTICAL for every client and every dataset)")
        return

    all_metrics, mi_aucs, expl = [], [], []
    for seed in cfg.seeds:
        print(f"\n=== seed {seed} | {cfg.dataset} | {cfg.model} | "
              f"DP={cfg.dp.enabled}(sigma={cfg.dp.noise_multiplier}) | "
              f"attack={cfg.attack.kind} ===")
        if args.centralized:
            model, m = run_centralized(cfg, seed)
            data = None
        else:
            import os as _os
            _os.makedirs("checkpoints", exist_ok=True)
            ckpt = _os.path.join("checkpoints",
                                 f"{args.name or 'run'}_{cfg.dataset}_"
                                 f"{cfg.model}_s{seed}.pkl")
            model, m, data = run_federated(cfg, seed, checkpoint_path=ckpt)
        print("  final:", {k: round(v, 4) for k, v in m.items()
                           if isinstance(v, float)})
        all_metrics.append(m)
        os.makedirs(cfg.results_dir, exist_ok=True)
        _pname = args.name or f"{cfg.dataset}_{cfg.model}"
        with open(os.path.join(cfg.results_dir, _pname + "_partial.json"),
                  "w") as fh:
            json.dump({"seeds_done": len(all_metrics),
                       "per_seed": all_metrics}, fh, indent=2)

        if data is not None:
            X_tr, X_te, y_tr, y_te, _ = data
            mi = membership_inference_auc(model, X_tr, y_tr, X_te, y_te,
                                          seed=seed)
            mi_aucs.append(mi)
            print(f"  MI attack AUC = {mi:.3f} (0.5 = no leakage)")

            if not args.skip_shap:
                import torch
                _dev = next(model.parameters()).device
                with torch.no_grad():
                    probs = torch.sigmoid(model(
                        torch.tensor(X_te[:5000]).to(_dev))).cpu().numpy()
                flagged = X_te[:5000][np.argsort(-probs)[:100]]
                phi, cov = dp_shap_for_flagged(model, flagged, X_tr,
                                               seed=seed)
                rho = global_rank_correlation(phi, X_tr[:20000], y_tr[:20000])
                expl.append({"coverage": cov, "spearman_rho": rho})
                print(f"  Explanation Coverage = {cov:.2f} | "
                      f"Spearman rho = {rho:.2f}")

    summary = {"config": {"dataset": cfg.dataset, "model": cfg.model,
                          "sigma": cfg.dp.noise_multiplier if cfg.dp.enabled
                          else None,
                          "T": cfg.fl.global_rounds, "E": cfg.fl.local_epochs,
                          "attack": cfg.attack.kind,
                          "aggregation": cfg.fl.aggregation,
                          "n_clients": cfg.fl.n_clients,
                          "k_trim": cfg.fl.k_trim},
               "n_seeds": len(cfg.seeds)}
    for key in ["f1", "auc", "pr_auc", "precision", "recall", "accuracy"]:
        vals = [m[key] for m in all_metrics]
        summary[key] = {"mean": float(np.mean(vals)),
                        "std": float(np.std(vals))}
    eps_vals = [m["epsilon_worst"] for m in all_metrics
                if m.get("epsilon_worst") is not None]
    summary["epsilon_worst_case"] = float(max(eps_vals)) if eps_vals else None
    if mi_aucs:
        summary["mi_attack_auc"] = {"mean": float(np.mean(mi_aucs)),
                                    "std": float(np.std(mi_aucs))}
    if expl:
        summary["explanation_coverage"] = {
            "mean": float(np.mean([e["coverage"] for e in expl])),
            "std": float(np.std([e["coverage"] for e in expl]))}
        rhos = [e["spearman_rho"] for e in expl
                if not np.isnan(e["spearman_rho"])]
        if rhos:
            summary["spearman_rho"] = {"mean": float(np.mean(rhos)),
                                       "std": float(np.std(rhos))}

    os.makedirs(cfg.results_dir, exist_ok=True)
    name = args.name or (f"{cfg.dataset}_{cfg.model}"
                         f"_{'dp' if cfg.dp.enabled else 'nodp'}"
                         f"_{cfg.attack.kind or 'benign'}")
    path = os.path.join(cfg.results_dir, name + ".json")
    with open(path, "w") as fh:
        json.dump({"summary": summary, "per_seed": all_metrics}, fh, indent=2)
    print(f"\nsaved -> {path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
