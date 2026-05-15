"""
X-PrivScore Dataset Pipeline
==============================
Handles loading, preprocessing, and federated partitioning for:
  1. IEEE-CIS Fraud Detection      (Kaggle)
  2. UCI Credit Card Fraud         (UCI ML Repository)
  3. Elliptic Bitcoin Dataset      (Kaggle / Harvard Dataverse)

Each dataset is:
  - Cleaned and normalized
  - Split into N non-IID federated client shards (Dirichlet)
  - Returned as PyTorch DataLoaders

Usage:
    from dataset_pipeline import get_federated_loaders
    clients = get_federated_loaders("uci", n_clients=10)
    train_loader, val_loader = clients[0]
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Base Transaction Dataset
# ─────────────────────────────────────────────
class TransactionDataset(Dataset):
    """Generic PyTorch Dataset for tabular fraud/risk data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# 2. Non-IID Federated Partitioner (Dirichlet)
# ─────────────────────────────────────────────
def dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Partition dataset indices into n_clients non-IID shards
    using Dirichlet distribution (standard FL heterogeneity).

    alpha=0.5  → moderate heterogeneity (used in paper)
    alpha=0.1  → high heterogeneity
    alpha=10.0 → near-IID
    """
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(labels))
    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]

    client_indices = [[] for _ in range(n_clients)]

    for cls_idx in class_indices:
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        # ensure no client gets 0 samples
        proportions = np.maximum(proportions, 1e-3)
        proportions /= proportions.sum()
        splits = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(cls_idx, splits)):
            client_indices[client_id].extend(chunk.tolist())

    return [np.array(idx) for idx in client_indices]


# ─────────────────────────────────────────────
# 3. UCI Credit Card Fraud Dataset
# ─────────────────────────────────────────────
def load_uci(data_path: str = "creditcard.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    UCI Credit Card Fraud Detection Dataset
    Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    File: creditcard.csv (~150 MB)

    Features: Time, V1-V28 (PCA), Amount → 30 features
    Label: Class (0=normal, 1=fraud)
    Size: 284,807 transactions, 0.172% fraud
    """
    print("[UCI] Loading dataset...")
    df = pd.read_csv(data_path)

    # Drop Time (not informative across clients)
    df = df.drop(columns=["Time"])

    # Log-transform Amount (heavy-tailed)
    df["Amount"] = np.log1p(df["Amount"])

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"[UCI] Shape: {X.shape}, Fraud rate: {y.mean()*100:.3f}%")
    return X, y


# ─────────────────────────────────────────────
# 4. IEEE-CIS Fraud Detection Dataset
# ─────────────────────────────────────────────
def load_ieee_cis(
    transaction_path: str = "train_transaction.csv",
    identity_path:    str = "train_identity.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    IEEE-CIS Fraud Detection Dataset
    Download: https://www.kaggle.com/competitions/ieee-fraud-detection/data
    Files: train_transaction.csv + train_identity.csv

    Features: ~400 raw → reduced to top features after selection
    Label: isFraud
    Size: ~590,000 transactions, ~3.5% fraud
    """
    print("[IEEE-CIS] Loading transaction data...")
    trans = pd.read_csv(transaction_path)

    # Optionally merge identity features
    if os.path.exists(identity_path):
        print("[IEEE-CIS] Merging identity features...")
        identity = pd.read_csv(identity_path)
        trans = trans.merge(identity, on="TransactionID", how="left")

    y = trans["isFraud"].values
    trans = trans.drop(columns=["TransactionID", "isFraud"])

    # Keep numeric columns only for simplicity
    trans = trans.select_dtypes(include=[np.number])

    # Fill missing with median
    trans = trans.fillna(trans.median())

    # Log-transform TransactionAmt
    if "TransactionAmt" in trans.columns:
        trans["TransactionAmt"] = np.log1p(trans["TransactionAmt"])

    # Feature selection: keep top 50 by variance
    variances = trans.var()
    top_features = variances.nlargest(50).index
    trans = trans[top_features]

    X = trans.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"[IEEE-CIS] Shape: {X.shape}, Fraud rate: {y.mean()*100:.3f}%")
    return X, y


# ─────────────────────────────────────────────
# 5. Elliptic Bitcoin Dataset
# ─────────────────────────────────────────────
def load_elliptic(
    features_path: str = "elliptic_txs_features.csv",
    classes_path:  str = "elliptic_txs_classes.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elliptic Bitcoin Transaction Dataset
    Download: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
    Files: elliptic_txs_features.csv + elliptic_txs_classes.csv

    Features: 166 (94 local + 72 aggregated neighborhood)
    Label: 1=illicit, 2=licit, unknown=removed
    Size: 203,769 transactions (46,564 labeled)

    WHY THIS DATASET?
    - Completely different domain (blockchain) → proves generalizability
    - Graph-structured → justifies attention mechanism in Tab-Transformer
    - Used in top-tier security papers → strong credibility with reviewers
    """
    print("[Elliptic] Loading dataset...")
    features = pd.read_csv(features_path, header=None)
    classes  = pd.read_csv(classes_path)

    # Merge on transaction ID (first column of features)
    features.columns = ["txId"] + [f"f{i}" for i in range(1, features.shape[1])]
    df = features.merge(classes, on="txId")

    # Remove unknown labels
    df = df[df["class"] != "unknown"].copy()

    # Convert labels: illicit=1, licit=0
    df["label"] = (df["class"] == "1").astype(int)

    # Drop ID and class columns
    X = df.drop(columns=["txId", "class", "label"]).values
    y = df["label"].values

    # Handle any remaining NaN
    X = np.nan_to_num(X, nan=0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"[Elliptic] Shape: {X.shape}, Illicit rate: {y.mean()*100:.3f}%")
    return X, y


# ─────────────────────────────────────────────
# 6. Federated DataLoader Builder
# ─────────────────────────────────────────────
class ClientLoaderList(list):
    """List subclass that allows storing test_loader as attribute."""
    pass


def get_federated_loaders(
    dataset_name: str,
    n_clients:    int   = 10,
    alpha:        float = 0.5,
    val_ratio:    float = 0.2,
    batch_size:   int   = 128,
    seed:         int   = 42,
    data_dir:     str   = "./data",
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Main entry point. Returns list of (train_loader, val_loader) per client.

    Args:
        dataset_name : "uci" | "ieee_cis" | "elliptic"
        n_clients    : number of federated clients
        alpha        : Dirichlet heterogeneity (0.5 = paper setting)
        val_ratio    : fraction of each client's data for local validation
        batch_size   : DataLoader batch size
        seed         : random seed for reproducibility

    Returns:
        List of length n_clients, each element is (train_loader, val_loader)

    Example:
        clients = get_federated_loaders("uci", n_clients=10)
        for client_id, (train_loader, val_loader) in enumerate(clients):
            # local training
    """
    # ── Load raw data ──────────────────────────
    loaders = {
        "uci":      lambda: load_uci(
            os.path.join(data_dir, "creditcard.csv")),
        "ieee_cis": lambda: load_ieee_cis(
            os.path.join(data_dir, "train_transaction.csv"),
            os.path.join(data_dir, "train_identity.csv")),
        "elliptic": lambda: load_elliptic(
            os.path.join(data_dir, "elliptic_txs_features.csv"),
            os.path.join(data_dir, "elliptic_txs_classes.csv")),
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from {list(loaders.keys())}")

    X, y = loaders[dataset_name]()

    # ── Global train/test split (keep test holdout) ─
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed)

    full_dataset = TransactionDataset(X_train, y_train)

    # ── Dirichlet non-IID partition ────────────
    client_idx = dirichlet_partition(y_train, n_clients, alpha, seed)

    # ── Build per-client DataLoaders ───────────
    client_loaders = ClientLoaderList()
    for cid, idx in enumerate(client_idx):
        n_val = max(1, int(len(idx) * val_ratio))
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=batch_size, shuffle=True,  drop_last=False)
        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=batch_size, shuffle=False, drop_last=False)

        client_loaders.append((train_loader, val_loader))
        print(f"  Client {cid:2d}: train={len(train_idx):6d}  "
              f"val={len(val_idx):5d}  "
              f"fraud_rate={y_train[idx].mean()*100:.2f}%")

    # ── Global test set ────────────────────────
    test_dataset = TransactionDataset(X_test, y_test)
    test_loader  = DataLoader(test_dataset, batch_size=512, shuffle=False)

    print(f"\n[{dataset_name.upper()}] "
          f"Total train: {len(X_train):,} | "
          f"Test: {len(X_test):,} | "
          f"Clients: {n_clients}")

    # Store test_loader as attribute on the list for easy access
    client_loaders.test_loader = test_loader  # type: ignore
    return client_loaders


# ─────────────────────────────────────────────
# 7. Dataset Statistics Reporter
# ─────────────────────────────────────────────
def print_dataset_stats(client_loaders: list, dataset_name: str):
    """Print per-client statistics — useful for paper's Table."""
    print(f"\n{'='*60}")
    print(f" {dataset_name.upper()} — Federated Partition Statistics")
    print(f"{'='*60}")
    print(f"{'Client':>8} {'Train':>8} {'Val':>6} {'Fraud%':>8}")
    print(f"{'-'*35}")
    for cid, (tl, vl) in enumerate(client_loaders):
        n_train  = len(tl.dataset)
        n_val    = len(vl.dataset)
        labels   = torch.cat([y for _, y in tl]).numpy()
        fraud_pct = labels.mean() * 100
        print(f"{cid:>8} {n_train:>8} {n_val:>6} {fraud_pct:>7.2f}%")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# 8. Download Instructions
# ─────────────────────────────────────────────
DOWNLOAD_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════╗
║           Dataset Download Instructions                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. UCI Credit Card Fraud                                    ║
║     URL : kaggle.com/datasets/mlg-ulb/creditcardfraud        ║
║     File: creditcard.csv  (~150 MB)                          ║
║     Cmd : kaggle datasets download mlg-ulb/creditcardfraud   ║
║                                                              ║
║  2. IEEE-CIS Fraud Detection                                 ║
║     URL : kaggle.com/competitions/ieee-fraud-detection       ║
║     Files: train_transaction.csv + train_identity.csv        ║
║     Cmd : kaggle competitions download ieee-fraud-detection  ║
║                                                              ║
║  3. Elliptic Bitcoin Dataset                                 ║
║     URL : kaggle.com/datasets/ellipticco/elliptic-data-set   ║
║     Files: elliptic_txs_features.csv                         ║
║             elliptic_txs_classes.csv                         ║
║     Cmd : kaggle datasets download ellipticco/elliptic-data-set║
║                                                              ║
║  Place all files in ./data/ directory.                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Install Kaggle CLI:
    pip install kaggle
    # Place kaggle.json in ~/.kaggle/
"""


# ─────────────────────────────────────────────
# 9. Quick Test (no real data needed)
# ─────────────────────────────────────────────
def _mock_test():
    """Test pipeline with synthetic data — no download needed."""
    print("Running mock test with synthetic data...\n")
    np.random.seed(42)
    n = 10_000
    X = np.random.randn(n, 13).astype(np.float32)
    y = (np.random.rand(n) < 0.016).astype(np.float32)  # 1.6% fraud

    dataset = TransactionDataset(X, y)
    idx     = dirichlet_partition(y.astype(int), n_clients=5, alpha=0.5)

    print("Client shard sizes and fraud rates:")
    for cid, i in enumerate(idx):
        fr = y[i].mean() * 100
        print(f"  Client {cid}: {len(i):5d} samples, fraud={fr:.2f}%")

    print("\n✅ Mock test passed. Ready for real datasets.")
    print(DOWNLOAD_INSTRUCTIONS)


if __name__ == "__main__":
    _mock_test()
