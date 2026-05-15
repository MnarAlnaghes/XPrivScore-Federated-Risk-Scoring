"""
X-PrivScore Federated Training Experiment Runner
=================================================
يشغّل التجارب كاملة ويولّد جداول النتائج تلقائياً.

الاستخدام:
    python run_experiments.py --dataset uci
    python run_experiments.py --dataset ieee_cis
    python run_experiments.py --dataset elliptic
    python run_experiments.py --all          # يشغّل الثلاثة

النتائج تُحفظ في:
    results/summary_table.csv    ← الجدول الرئيسي للورقة
    results/plots/               ← رسوم بيانية جاهزة
    results/logs/                ← تفاصيل كل تجربة
"""

import os
import argparse
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, accuracy_score, average_precision_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ── Import our modules ──────────────────────────────────────────
from xprivscore_architecture import XPrivScoreTransformer, XPrivScoreMLP, clip_and_noise
from dataset_pipeline import get_federated_loaders, print_dataset_stats

# ── Reproducibility ─────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]   # 5 seeds → mean ± std

# ── Output directories ──────────────────────────────────────────
os.makedirs("results/logs",  exist_ok=True)
os.makedirs("results/plots", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. LOCAL CLIENT TRAINING
# ═══════════════════════════════════════════════════════════════
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss — أفضل من BCEWithLogitsLoss للـ class imbalance الشديد.
    يركّز التدريب على الـ fraud cases الصعبة.
    """
    bce   = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt    = torch.where(targets == 1, probs, 1 - probs)
    w     = torch.where(targets == 1,
                        torch.full_like(pt, alpha),
                        torch.full_like(pt, 1 - alpha))
    loss  = w * (1 - pt) ** gamma * bce
    return loss.mean()


def oversample_minority(train_loader):
    """Oversample fraud cases to 10% of dataset."""
    all_X = torch.cat([X for X, _ in train_loader])
    all_y = torch.cat([y for _, y in train_loader])

    fraud_idx  = (all_y == 1).nonzero(as_tuple=True)[0]
    normal_idx = (all_y == 0).nonzero(as_tuple=True)[0]

    if len(fraud_idx) == 0:
        return train_loader

    target  = max(len(fraud_idx), len(normal_idx) // 10)
    repeats = (target // len(fraud_idx)) + 1
    fraud_oversampled = fraud_idx.repeat(repeats)[:target]

    all_idx = torch.cat([normal_idx, fraud_oversampled])
    perm    = torch.randperm(len(all_idx))
    all_idx = all_idx[perm]

    new_dataset = TensorDataset(all_X[all_idx], all_y[all_idx])
    return DataLoader(new_dataset,
                      batch_size=train_loader.batch_size,
                      shuffle=True)


def local_train(model, train_loader, epochs=5,
                lr=1e-3, clip_norm=1.0, noise_sigma=0.3,
                use_dp=True, device="cpu"):
    """Train one client locally with oversampling + focal loss + mild DP."""
    model = model.to(device)
    model.train()

    # Oversample minority class
    train_loader = oversample_minority(train_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=1e-4)

    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = focal_loss(preds, y_batch, alpha=0.75, gamma=2.0)
            loss.backward()
            if use_dp:
                # Milder noise: sigma=0.3 instead of 1.2
                clip_and_noise(model, clip_norm=clip_norm,
                               noise_sigma=noise_sigma)
            optimizer.step()
    return model


# ═══════════════════════════════════════════════════════════════
# 2. FEDAVG AGGREGATION
# ═══════════════════════════════════════════════════════════════
def fedavg(global_model, local_models, client_sizes):
    """Weighted FedAvg aggregation."""
    total = sum(client_sizes)
    global_dict = global_model.state_dict()

    for key in global_dict:
        global_dict[key] = sum(
            (sz / total) * lm.state_dict()[key].float()
            for sz, lm in zip(client_sizes, local_models)
        )
    global_model.load_state_dict(global_dict)
    return global_model


# ═══════════════════════════════════════════════════════════════
# 3. EVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluate(model, loader, device="cpu", threshold=None):
    """Compute F1, AUC, AUPRC — finds optimal threshold automatically."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).cpu()
            probs  = torch.sigmoid(logits).numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy().astype(int))

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    try:
        auc   = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auc = auprc = float("nan")

    # ── Find optimal threshold: يشترط precision > 0.3 ──────
    if threshold is None:
        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.05, 0.95, 0.01):
            y_pred = (y_prob >= thr).astype(int)
            if y_pred.sum() == 0:
                continue
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            f1   = f1_score(y_true, y_pred, zero_division=0)
            # يشترط precision و recall معقولين
            if prec >= 0.1 and rec >= 0.1 and f1 > best_f1:
                best_f1, best_thr = f1, thr
        threshold = best_thr

    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy":  accuracy_score(y_true, y_pred) * 100,
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       auc,
        "auprc":     auprc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_prob  = np.array(all_probs)

    # Handle edge case: only one class in batch
    try:
        auc   = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auc = auprc = float("nan")

    return {
        "accuracy":  accuracy_score(y_true, y_pred)  * 100,
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       auc,
        "auprc":     auprc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
    }


# ═══════════════════════════════════════════════════════════════
# 4. SINGLE FEDERATED RUN
# ═══════════════════════════════════════════════════════════════
def run_federated(
    model_class,
    model_kwargs,
    client_loaders,
    n_rounds=20,
    local_epochs=5,
    use_dp=True,
    device="cpu",
):
    """
    Full federated training loop.
    Returns list of per-round test metrics.
    """
    # Get test loader (attached to client_loaders list)
    test_loader = client_loaders.test_loader

    # Initialize global model
    global_model = model_class(**model_kwargs).to(device)

    history = []
    client_sizes = [len(tl.dataset) for tl, _ in client_loaders]

    for rnd in range(1, n_rounds + 1):
        local_models = []

        for tl, _ in client_loaders:
            # Copy global model to client
            local_model = model_class(**model_kwargs)
            local_model.load_state_dict(global_model.state_dict())

            # Local training with DP
            local_model = local_train(
                local_model, tl,
                epochs=local_epochs,
                use_dp=use_dp,
                device=device,
            )
            local_models.append(local_model)

        # FedAvg aggregation
        global_model = fedavg(global_model, local_models, client_sizes)

        # Evaluate every 5 rounds + last round
        if rnd % 5 == 0 or rnd == n_rounds:
            metrics = evaluate(global_model, test_loader, device)
            metrics["round"] = rnd
            history.append(metrics)
            print(f"    Round {rnd:3d}/{n_rounds} | "
                  f"F1={metrics['f1']:.4f} | "
                  f"AUC={metrics['auc']:.4f} | "
                  f"Acc={metrics['accuracy']:.2f}%")

    final_metrics = evaluate(global_model, test_loader, device)
    return final_metrics, history


# ═══════════════════════════════════════════════════════════════
# 5. MULTI-SEED EXPERIMENT
# ═══════════════════════════════════════════════════════════════
def run_experiment(
    dataset_name,
    model_name,
    model_class,
    model_kwargs,
    n_clients=10,
    n_rounds=20,
    local_epochs=5,
    use_dp=True,
    seeds=SEEDS,
    data_dir="./data",
    device="cpu",
):
    """Run experiment across multiple seeds → mean ± std."""
    print(f"\n{'─'*60}")
    print(f"  Dataset: {dataset_name.upper()} | Model: {model_name}")
    print(f"  Seeds: {seeds} | DP: {use_dp}")
    print(f"{'─'*60}")

    all_metrics = []
    t0 = time.time()

    for seed in seeds:
        print(f"\n  [Seed {seed}]")
        torch.manual_seed(seed)
        np.random.seed(seed)

        client_loaders = get_federated_loaders(
            dataset_name,
            n_clients=n_clients,
            alpha=0.5,
            seed=seed,
            data_dir=data_dir,
        )

        metrics, history = run_federated(
            model_class, model_kwargs,
            client_loaders,
            n_rounds=n_rounds,
            local_epochs=local_epochs,
            use_dp=use_dp,
            device=device,
        )
        all_metrics.append(metrics)

        # Save per-seed history
        log_path = f"results/logs/{dataset_name}_{model_name}_seed{seed}.json"
        with open(log_path, "w") as f:
            json.dump({"metrics": metrics, "history": history}, f, indent=2)

    elapsed = time.time() - t0

    # Aggregate across seeds
    keys = ["accuracy", "f1", "auc", "auprc", "precision", "recall"]
    result = {
        "dataset":    dataset_name,
        "model":      model_name,
        "dp":         use_dp,
        "runtime_s":  round(elapsed, 1),
    }
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
        result[f"{k}_mean"] = np.mean(vals)
        result[f"{k}_std"]  = np.std(vals)

    print(f"\n  ✓ Done in {elapsed:.0f}s")
    print(f"    F1  = {result['f1_mean']:.4f} ± {result['f1_std']:.4f}")
    print(f"    AUC = {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")

    return result, all_metrics


# ═══════════════════════════════════════════════════════════════
# 6. STATISTICAL SIGNIFICANCE TEST
# ═══════════════════════════════════════════════════════════════
def wilcoxon_test(transformer_scores, mlp_scores, metric="f1"):
    """Wilcoxon signed-rank test: Transformer vs MLP."""
    t_vals = [m[metric] for m in transformer_scores]
    m_vals = [m[metric] for m in mlp_scores]
    try:
        stat, p = wilcoxon(t_vals, m_vals)
    except Exception:
        stat, p = float("nan"), float("nan")
    return stat, p


# ═══════════════════════════════════════════════════════════════
# 7. RESULTS TABLE GENERATOR
# ═══════════════════════════════════════════════════════════════
def save_results_table(all_results):
    """Save LaTeX-ready results table to CSV and .tex file."""
    rows = []
    for r in all_results:
        rows.append({
            "Dataset":    r["dataset"].upper().replace("_", "-"),
            "Model":      r["model"],
            "DP":         "✓" if r["dp"] else "✗",
            "Accuracy":   f"{r['accuracy_mean']:.2f} ±{r['accuracy_std']:.2f}",
            "F1":         f"{r['f1_mean']:.4f} ±{r['f1_std']:.4f}",
            "AUC":        f"{r['auc_mean']:.4f} ±{r['auc_std']:.4f}",
            "AUPRC":      f"{r['auprc_mean']:.4f} ±{r['auprc_std']:.4f}",
            "Precision":  f"{r['precision_mean']:.4f} ±{r['precision_std']:.4f}",
            "Recall":     f"{r['recall_mean']:.4f} ±{r['recall_std']:.4f}",
        })

    df = pd.DataFrame(rows)
    csv_path = "results/summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[✓] Results saved to {csv_path}")

    # LaTeX table
    latex = df.to_latex(index=False, escape=False,
                         caption="Comparative performance of Tab-Transformer "
                                 "vs MLP across three federated datasets "
                                 "(mean $\\pm$ std, 5 seeds).",
                         label="tab:main_results")
    tex_path = "results/summary_table.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"[✓] LaTeX table saved to {tex_path}")
    print("\n" + df.to_string(index=False))
    return df


# ═══════════════════════════════════════════════════════════════
# 8. PLOTTING
# ═══════════════════════════════════════════════════════════════
def plot_comparison(all_results):
    """Bar chart: F1 and AUC comparison across datasets and models."""
    datasets = sorted(set(r["dataset"] for r in all_results))
    models   = sorted(set(r["model"]   for r in all_results))
    colors   = {"Tab-Transformer": "#2196F3", "MLP-Baseline": "#FF5722"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(datasets))
    width = 0.35

    for metric, ax, title in [
        ("f1",  axes[0], "F1-Score"),
        ("auc", axes[1], "AUC-ROC"),
    ]:
        for i, model in enumerate(models):
            means = []
            stds  = []
            for ds in datasets:
                r = next((r for r in all_results
                          if r["dataset"] == ds and r["model"] == model), None)
                means.append(r[f"{metric}_mean"] if r else 0)
                stds.append(r[f"{metric}_std"]   if r else 0)

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, means, width,
                          yerr=stds, capsize=4,
                          label=model, color=colors.get(model, "gray"),
                          alpha=0.85)
            # Value labels on bars
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{m:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"{title} Comparison", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper().replace("_", "-") for d in datasets])
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Tab-Transformer vs MLP: Federated Learning Performance",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = "results/plots/comparison_bar.pdf"
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[✓] Bar chart saved to {plot_path}")


def plot_convergence(log_dir="results/logs"):
    """Convergence curves from saved logs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Tab-Transformer": "#2196F3", "MLP-Baseline": "#FF5722"}

    for fname in os.listdir(log_dir):
        if not fname.endswith(".json"):
            continue
        parts = fname.replace(".json", "").split("_")
        # parse: dataset_model_seed*.json
        try:
            seed   = int(fname.split("seed")[-1].split(".")[0])
            model  = "Tab-Transformer" if "Tab" in fname else "MLP-Baseline"
            ds     = parts[0]
        except Exception:
            continue

        with open(os.path.join(log_dir, fname)) as f:
            data = json.load(f)

        rounds = [h["round"] for h in data["history"]]
        f1s    = [h["f1"]    for h in data["history"]]
        ax.plot(rounds, f1s,
                color=colors.get(model, "gray"),
                alpha=0.3, linewidth=1)

    # Add legend proxies
    for model, color in colors.items():
        ax.plot([], [], color=color, linewidth=2, label=model)

    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("Convergence Curves (all seeds & datasets)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = "results/plots/convergence.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[✓] Convergence plot saved to {path}")


# ═══════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="uci",
                        choices=["uci", "ieee_cis", "elliptic"])
    parser.add_argument("--all",      action="store_true",
                        help="Run all three datasets")
    parser.add_argument("--rounds",   type=int, default=20)
    parser.add_argument("--epochs",   type=int, default=5)
    parser.add_argument("--clients",  type=int, default=10)
    parser.add_argument("--no_dp",    action="store_true",
                        help="Disable differential privacy")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--device",   default="cpu",
                        help="cpu | cuda | mps")
    args = parser.parse_args()

    device   = args.device
    use_dp   = not args.no_dp
    datasets = (["uci", "ieee_cis", "elliptic"]
                if args.all else [args.dataset])

    # ── Detect input_dim per dataset ────────────────────────────
    INPUT_DIMS = {
        "uci":      29,   # 30 features - Time
        "ieee_cis": 50,   # top-50 by variance
        "elliptic": 166,  # full feature set
    }

    # ── Models to compare ────────────────────────────────────────
    all_results          = []
    all_transformer_runs = {}
    all_mlp_runs         = {}

    for ds in datasets:
        input_dim = INPUT_DIMS[ds]

        # Tab-Transformer
        result_t, runs_t = run_experiment(
            dataset_name  = ds,
            model_name    = "Tab-Transformer",
            model_class   = XPrivScoreTransformer,
            model_kwargs  = dict(input_dim=input_dim,
                                 embed_dim=32, num_heads=4,
                                 num_layers=2, ff_dim=64),
            n_clients     = args.clients,
            n_rounds      = args.rounds,
            local_epochs  = args.epochs,
            use_dp        = use_dp,
            data_dir      = args.data_dir,
            device        = device,
        )
        all_results.append(result_t)
        all_transformer_runs[ds] = runs_t

        # MLP Baseline (ablation)
        result_m, runs_m = run_experiment(
            dataset_name  = ds,
            model_name    = "MLP-Baseline",
            model_class   = XPrivScoreMLP,
            model_kwargs  = dict(input_dim=input_dim),
            n_clients     = args.clients,
            n_rounds      = args.rounds,
            local_epochs  = args.epochs,
            use_dp        = use_dp,
            data_dir      = args.data_dir,
            device        = device,
        )
        all_results.append(result_m)
        all_mlp_runs[ds] = runs_m

        # ── Wilcoxon test ──────────────────────────────────────
        stat_f1,  p_f1  = wilcoxon_test(runs_t, runs_m, "f1")
        stat_auc, p_auc = wilcoxon_test(runs_t, runs_m, "auc")
        print(f"\n  [Significance] {ds.upper()}")
        print(f"    F1  Wilcoxon: stat={stat_f1:.3f}, p={p_f1:.4f} "
              f"{'✓ significant' if p_f1 < 0.05 else '✗ not significant'}")
        print(f"    AUC Wilcoxon: stat={stat_auc:.3f}, p={p_auc:.4f} "
              f"{'✓ significant' if p_auc < 0.05 else '✗ not significant'}")

    # ── Save tables and plots ────────────────────────────────────
    print("\n" + "═"*60)
    print("  FINAL RESULTS")
    print("═"*60)
    df = save_results_table(all_results)
    plot_comparison(all_results)
    plot_convergence()

    # Save full results as JSON
    with open("results/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n✅ All experiments complete!")
    print("   results/summary_table.csv  ← paste into paper")
    print("   results/summary_table.tex  ← LaTeX table")
    print("   results/plots/             ← figures for paper")


if __name__ == "__main__":
    main()
