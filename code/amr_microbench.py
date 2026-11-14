"""AMR micro-benchmarks at the paper's main configuration (N=10).

Produces results/amr_micro.json, the source for:
  * maximum mask-cancellation error over 1,000 random aggregations of
    18k-dimensional updates (Sec. V-A, IX-J),
  * the collusion Monte-Carlo at N=10, k=5, R=6 over 1e5 and 1e6 trials
    (Sec. V-B, IX-J),
  * Table III (per-round aggregation latency at d=4,000 and d=18,000).

Correctness and collusion are deterministic given the seed. Latency is
wall-clock and depends on the hardware; rerunning it reproduces the ratio
(AMR / pairwise) rather than the absolute milliseconds.

Run from the repository root:
    python code/amr_microbench.py
"""
import json
import os

import hashlib

import numpy as np

from amr import amr_aggregate, collusion_simulation, latency_benchmark


def main():
    # correctness: maximum error over 1,000 independent random aggregations
    key = hashlib.sha256(b"shared-seed").digest()
    worst = 0.0
    for s in range(1_000):
        rng = np.random.default_rng(s)
        ups = [rng.normal(size=18_000) for _ in range(10)]
        agg, _ = amr_aggregate(ups, key, round_t=s)
        worst = max(worst, float(np.max(np.abs(agg - np.sum(ups, axis=0)))))
    out = {
        "correctness_max_abs_error_d18000_1000_aggregations": worst,
        "collusion_N10_k5_R6": {
            "trials": 100_000,
            "seed": 0,
            "empirical": collusion_simulation(n_clients=10, k_colluders=5,
                                              rounds_R=6, trials=100_000,
                                              seed=0),
            "bound_(k/N)^R": (5 / 10) ** 6,
        },
        "collusion_N10_k5_R6_1e6": {
            "trials": 1_000_000,
            "seed": 1,
            "empirical": collusion_simulation(n_clients=10, k_colluders=5,
                                              rounds_R=6, trials=1_000_000,
                                              seed=1),
        },
        "latency_ms": {},
    }
    for dim in (4_000, 18_000):
        r = latency_benchmark(dim=dim, n=10, rounds=20, seed=0)
        out["latency_ms"][str(dim)] = {
            "amr": 1000 * r["amr_mean_s"],
            "pairwise": 1000 * r["bonawitz_mean_s"],
            "ratio": r["latency_ratio_amr_over_bonawitz"],
        }
    os.makedirs("results", exist_ok=True)
    with open("results/amr_micro.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print("saved -> results/amr_micro.json")


if __name__ == "__main__":
    main()
