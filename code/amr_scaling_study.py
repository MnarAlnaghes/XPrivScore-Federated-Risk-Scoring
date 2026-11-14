"""AMR scalability and rotation-depth study for the extended paper.

Generates three deterministic (CPU, fast) result sets:
  1. Collusion bound vs. R at several (N, k): empirical success vs (k/N)^R.
  2. Latency scaling with N: AMR vs pairwise baseline as N grows.
  3. Correctness at scale: mask-cancellation error vs N.

All results are written to results/amr_scaling.json for the paper.
"""
import json
import os
import numpy as np
from amr import (collusion_simulation, latency_benchmark, verify_correctness,
                 amr_aggregate, bonawitz_aggregate)
import hashlib
import time


def collusion_vs_R(configs, trials=200_000):
    """Empirical reconstruction success vs theoretical (k/N)^R."""
    rows = []
    for N, k in configs:
        for R in [2, 4, 6, 8, 10]:
            emp = collusion_simulation(n_clients=N, k_colluders=k,
                                       rounds_R=R, trials=trials)
            theo = (k / N) ** R
            rows.append({"N": N, "k": k, "R": R,
                         "empirical": emp, "theory": theo,
                         "k_over_N": k / N})
            print(f"  N={N} k={k} R={R}: emp={emp:.5f} theory={theo:.5f}")
    return rows


def latency_vs_N(Ns, dim=4000, rounds=20):
    """AMR vs pairwise latency as the number of clients grows."""
    rows = []
    for N in Ns:
        res = latency_benchmark(dim=dim, n=N, rounds=rounds)
        rows.append({"N": N, "dim": dim,
                     "amr_ms": res["amr_mean_s"] * 1000,
                     "bonawitz_ms": res["bonawitz_mean_s"] * 1000,
                     "ratio": res["latency_ratio_amr_over_bonawitz"]})
        print(f"  N={N}: AMR={res['amr_mean_s']*1000:.2f}ms "
              f"pairwise={res['bonawitz_mean_s']*1000:.2f}ms "
              f"ratio={res['latency_ratio_amr_over_bonawitz']:.3f}")
    return rows


def correctness_vs_N(Ns, dim=4000):
    """Max mask-cancellation error as N grows (should stay ~1e-12)."""
    rows = []
    rng = np.random.default_rng(0)
    key = hashlib.sha256(b"scaling-seed").digest()
    for N in Ns:
        ups = [rng.normal(size=dim) for _ in range(N)]
        amr_sum, _ = amr_aggregate(ups, key, round_t=3)
        err = float(np.max(np.abs(amr_sum - np.sum(ups, axis=0))))
        rows.append({"N": N, "max_error": err})
        print(f"  N={N}: max cancellation error {err:.2e}")
    return rows


if __name__ == "__main__":
    print("1) Collusion bound vs R (validates Lemma 1 across N,k,R)...")
    collusion = collusion_vs_R([(10, 5), (20, 10), (50, 25), (100, 50)])

    print("2) Latency scaling with N...")
    latency = latency_vs_N([10, 20, 50, 100, 200])

    print("3) Correctness at scale...")
    correctness = correctness_vs_N([10, 20, 50, 100, 200])

    os.makedirs("results", exist_ok=True)
    with open("results/amr_scaling.json", "w") as f:
        json.dump({"collusion_vs_R": collusion,
                   "latency_vs_N": latency,
                   "correctness_vs_N": correctness}, f, indent=2)
    print("\nsaved -> results/amr_scaling.json")
