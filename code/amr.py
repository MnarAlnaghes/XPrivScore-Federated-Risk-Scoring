"""Adaptive Mask Rotation (AMR) — corrected construction.

Design (matches the revised paper Sec. 5.2):
  * masks are FRESHLY DERIVED every round from a shared seed:
        r_i^(t) = PRF_K(t || i)          (never reused -> no one-time-pad reuse)
  * client pairing is a pseudorandom PERMUTATION per round:
        sigma_t = PRP_K(t)               (guarantees exact cancellation)
  * client i uploads   g_i + r_i^(t) - r_{sigma_t(i)}^(t)
  * sum over all clients cancels masks exactly (correctness).

Security intuition (revised Lemma 1): reconstructing client i's update in
round t requires knowing r_{sigma_t(i)}^(t); across R rounds the counterparts
are pseudorandom, so k colluders succeed w.p. <= (k/N)^R + negl(lambda).

Baseline for latency comparison: Bonawitz-style pairwise masking, where every
round each pair (i,j) derives a fresh pairwise secret (simulated per-pair
key-agreement cost), giving O(N^2) setup per round vs AMR's O(N).
"""
import hashlib
import hmac
import time
import numpy as np


def _prf_bytes(key: bytes, msg: bytes, n_bytes: int) -> bytes:
    out, counter = b"", 0
    while len(out) < n_bytes:
        out += hmac.new(key, msg + counter.to_bytes(4, "big"),
                        hashlib.sha256).digest()
        counter += 1
    return out[:n_bytes]


def prf_mask(key: bytes, round_t: int, client_i: int, dim: int,
             scale: float = 1e3) -> np.ndarray:
    """Fresh pseudorandom mask r_i^(t) in R^dim derived from the shared seed."""
    raw = _prf_bytes(key, f"mask|{round_t}|{client_i}".encode(), 4 * dim)
    ints = np.frombuffer(raw, dtype=np.uint32).astype(np.float64)
    return ((ints / 2**32) - 0.5) * scale


def prp_permutation(key: bytes, round_t: int, n: int) -> np.ndarray:
    """sigma_t: keyed pseudorandom permutation of [n] (Fisher-Yates on PRF)."""
    seed_bytes = _prf_bytes(key, f"perm|{round_t}".encode(), 8)
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
    perm = rng.permutation(n)
    # avoid fixed points where a client masks with itself (degenerate)
    for i in np.where(perm == np.arange(n))[0]:
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def amr_aggregate(updates: list, key: bytes, round_t: int):
    """Mask updates with AMR, sum at the 'aggregator', verify cancellation.

    Returns (summed_update, wall_time_seconds).
    """
    n, dim = len(updates), updates[0].size
    t0 = time.perf_counter()
    sigma = prp_permutation(key, round_t, n)
    masks = [prf_mask(key, round_t, i, dim) for i in range(n)]
    masked = [u.reshape(-1) + masks[i] - masks[sigma[i]]
              for i, u in enumerate(updates)]
    total = np.sum(masked, axis=0)
    elapsed = time.perf_counter() - t0
    return total.reshape(updates[0].shape), elapsed


def bonawitz_aggregate(updates: list, key: bytes, round_t: int,
                       ka_cost_s: float = 2e-4):
    """Bonawitz-style pairwise masking baseline.

    Every unordered pair (i<j) derives a fresh pairwise mask each round;
    ka_cost_s simulates the per-pair key-agreement cost (ECDH ~0.2 ms).
    """
    n, dim = len(updates), updates[0].size
    t0 = time.perf_counter()
    masked = [u.reshape(-1).astype(np.float64).copy() for u in updates]
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            m = prf_mask(key, round_t, i * 10_000 + j, dim)
            masked[i] += m
            masked[j] -= m
            n_pairs += 1
    total = np.sum(masked, axis=0)
    elapsed = time.perf_counter() - t0 + n_pairs * ka_cost_s
    return total.reshape(updates[0].shape), elapsed


def verify_correctness(dim: int = 1000, n: int = 10, seed: int = 0) -> float:
    """Max abs error between plain sum and AMR-masked sum (should be ~1e-9)."""
    rng = np.random.default_rng(seed)
    ups = [rng.normal(size=dim) for _ in range(n)]
    key = hashlib.sha256(b"shared-seed").digest()
    amr_sum, _ = amr_aggregate(ups, key, round_t=7)
    return float(np.max(np.abs(amr_sum - np.sum(ups, axis=0))))


def collusion_simulation(n_clients=10, k_colluders=5, rounds_R=6,
                         trials=100_000, seed=0) -> float:
    """Empirical probability that ALL R counterparts of a target client are
    colluders (the event enabling reconstruction). Theory: <= (k/N)^R."""
    rng = np.random.default_rng(seed)
    colluders = set(range(k_colluders))
    target = n_clients - 1
    wins = 0
    for _ in range(trials):
        ok = True
        for _r in range(rounds_R):
            perm = rng.permutation(n_clients)
            if perm[target] not in colluders:
                ok = False
                break
        wins += ok
    return wins / trials


def latency_benchmark(dim=18_000, n=10, rounds=20, seed=0):
    """Mean per-round aggregation wall time: AMR vs Bonawitz baseline."""
    rng = np.random.default_rng(seed)
    key = hashlib.sha256(b"shared-seed").digest()
    ups = [rng.normal(size=dim) for _ in range(n)]
    t_amr, t_bon = [], []
    for t in range(rounds):
        _, e1 = amr_aggregate(ups, key, t)
        _, e2 = bonawitz_aggregate(ups, key, t)
        t_amr.append(e1)
        t_bon.append(e2)
    return {"amr_mean_s": float(np.mean(t_amr)),
            "bonawitz_mean_s": float(np.mean(t_bon)),
            "latency_ratio_amr_over_bonawitz":
                float(np.mean(t_amr) / np.mean(t_bon))}
