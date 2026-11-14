"""Server-side aggregation rules and client-side attacks.

Aggregation operates on flattened update vectors (one per client).
"""
import numpy as np


def weighted_fedavg(updates: list, weights: list) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return np.sum([wi * u for wi, u in zip(w, updates)], axis=0)


def trimmed_mean(updates: list, k_trim: int) -> np.ndarray:
    """Coordinate-wise trimmed mean: drop k largest and k smallest per coord."""
    U = np.stack(updates)                       # (N, dim)
    n = U.shape[0]
    if 2 * k_trim >= n:
        raise ValueError("2*k_trim must be < number of clients")
    S = np.sort(U, axis=0)
    return S[k_trim:n - k_trim].mean(axis=0)


# ── attacks (applied to a malicious client's data or update) ────────────────

def flip_labels(y: np.ndarray, frac: float, rng) -> np.ndarray:
    y = y.copy()
    idx = rng.choice(len(y), size=int(frac * len(y)), replace=False)
    y[idx] = 1 - y[idx]
    return y


def scale_update(update: np.ndarray, factor: float) -> np.ndarray:
    """Gradient rescaling attack: update * factor (e.g. -2x, -3x sign-flip)."""
    return update * factor


def collusive_update(update: np.ndarray, target_direction: np.ndarray,
                     strength: float = 3.0) -> np.ndarray:
    """Coordinated colluders push toward a common target direction."""
    d = target_direction / (np.linalg.norm(target_direction) + 1e-12)
    return update + strength * np.linalg.norm(update) * d
