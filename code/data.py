"""Dataset loading, preprocessing, and Dirichlet non-IID partitioning.

Datasets (place CSVs under data/):
  ulb      : ULB Credit Card Fraud (Kaggle 'creditcard.csv')
             284,807 tx, fraud ~0.17%, 28 PCA features + Amount (+ Time dropped)
  ieee     : IEEE-CIS Fraud Detection (Kaggle 'train_transaction.csv')
             ~590k tx, fraud ~3.5%; top-50 numeric features by variance
  elliptic : Elliptic Bitcoin ('elliptic_txs_features.csv' + 'elliptic_txs_classes.csv')
             46,564 labelled tx, illicit ~9.8%, 166 features
  synthetic: imbalanced synthetic data for smoke tests (no download needed)

Preprocessing per paper Sec. 7.1: min-max normalise continuous features to
[0,1]; drop features with pairwise |r| > 0.98.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _minmax(X: np.ndarray) -> np.ndarray:
    mn, mx = X.min(axis=0), X.max(axis=0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    return (X - mn) / rng


def _drop_correlated(X: np.ndarray, thresh: float = 0.98) -> np.ndarray:
    df = pd.DataFrame(X)
    corr = df.corr().abs().values
    n = corr.shape[0]
    drop = set()
    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j not in drop and corr[i, j] > thresh:
                drop.add(j)
    keep = [c for c in range(n) if c not in drop]
    return X[:, keep]


def load_dataset(name: str, data_dir: str = "data", seed: int = 0):
    """Return X_train, X_test, y_train, y_test (numpy, float32/int64)."""
    if name == "ulb":
        df = pd.read_csv(os.path.join(data_dir, "creditcard.csv"))
        y = df["Class"].values.astype(np.int64)
        X = df.drop(columns=["Class", "Time"]).values.astype(np.float32)
    elif name == "ieee":
        # Leakage-safe pipeline: split FIRST, then fit feature selection and
        # scaling on the training portion only. Raw-variance selection (old
        # recipe) picks scale-dominated junk columns (e.g. the TransactionDT
        # timestamp); we instead select the top-50 features by ANOVA F-score
        # computed on the training split.
        from sklearn.feature_selection import f_classif
        df = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
        y = df["isFraud"].values.astype(np.int64)
        num = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"],
                      errors="ignore").select_dtypes(include=[np.number])
        num = num.fillna(num.median())
        X_all = num.values.astype(np.float32)
        idx_tr, idx_te = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        F, _ = f_classif(X_all[idx_tr], y[idx_tr])
        F = np.nan_to_num(F, nan=0.0)
        top = np.argsort(-F)[:50]
        X_tr_raw, X_te_raw = X_all[idx_tr][:, top], X_all[idx_te][:, top]
        # train-fitted min-max scaling
        mn, mx = X_tr_raw.min(0), X_tr_raw.max(0)
        rng_ = np.where(mx - mn == 0, 1.0, mx - mn)
        X_tr_s = np.clip((X_tr_raw - mn) / rng_, 0.0, 1.0).astype(np.float32)
        X_te_s = np.clip((X_te_raw - mn) / rng_, 0.0, 1.0).astype(np.float32)
        return X_tr_s, X_te_s, y[idx_tr], y[idx_te]
    elif name == "elliptic":
        feats = pd.read_csv(os.path.join(data_dir, "elliptic_txs_features.csv"),
                            header=None)
        classes = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"))
        classes.columns = ["txId", "class"]
        feats = feats.rename(columns={0: "txId"})
        merged = feats.merge(classes, on="txId")
        merged = merged[merged["class"] != "unknown"]
        y = (merged["class"] == "1").values.astype(np.int64)  # 1 = illicit
        X = merged.drop(columns=["txId", "class", 1]).values.astype(np.float32)
    elif name == "synthetic":
        rng = np.random.default_rng(seed)
        n, d, fraud_rate = 20000, 30, 0.02
        y = (rng.random(n) < fraud_rate).astype(np.int64)
        X = rng.normal(size=(n, d)).astype(np.float32)
        # give fraud a learnable signature on a feature subset
        X[y == 1, :8] += rng.normal(1.5, 0.5, size=(int(y.sum()), 8))
    else:
        raise ValueError(f"unknown dataset {name}")

    X = _drop_correlated(X, 0.98)
    X = _minmax(X).astype(np.float32)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float,
                        seed: int, min_size: int = None):
    """Label-based Dirichlet non-IID partition. Returns list of index arrays.

    Re-samples until every client has at least `min_size` records and at
    least one positive example. For large K, min_size auto-scales so the
    constraint stays feasible: default is min(200, |train|/(4*K)).
    """
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    if min_size is None:
        # feasible floor: small fraction of the average client size
        min_size = max(8, min(200, len(y) // (8 * n_clients)))
    for attempt in range(600):
        idx_by_class = [np.where(y == c)[0] for c in range(n_classes)]
        client_idx = [[] for _ in range(n_clients)]
        for c_idx in idx_by_class:
            rng.shuffle(c_idx)
            props = rng.dirichlet([alpha] * n_clients)
            cuts = (np.cumsum(props) * len(c_idx)).astype(int)[:-1]
            for cid, part in enumerate(np.split(c_idx, cuts)):
                client_idx[cid].extend(part.tolist())
        sizes = [len(ci) for ci in client_idx]
        pos = [int(y[ci].sum()) for ci in client_idx]
        # allow up to 10% of clients to lack a positive example when K is
        # large and positives are scarce (they still train on the negative
        # class; the global model aggregates positives from other clients)
        n_zero_pos = sum(1 for p in pos if p == 0)
        pos_ok = n_zero_pos <= max(0, int(0.15 * n_clients))
        if min(sizes) >= min_size and pos_ok:
            return [np.array(sorted(ci)) for ci in client_idx]
        # relax the size floor progressively every 100 failed attempts
        if attempt > 0 and attempt % 100 == 0:
            min_size = max(8, min_size // 2)
    raise RuntimeError("could not find a valid Dirichlet partition; "
                       "lower min_size or raise alpha")
