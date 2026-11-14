"""Table V: worst-case client budget under fixed-batch DP-SGD vs uniform-q.

Fixed batch B gives client k the sampling rate q_k = B/|D_k| and
E*ceil(|D_k|/B) noisy steps per round, so the smallest client pays most.
Uniform-q gives every client the same (q, steps) and hence the same epsilon.

Smallest-client sizes are those of the paper's Dirichlet(0.5) partitions
(K=10, seed 0). Accountant: RDP with Opacus' default orders.

Run from the repository root:
    python code/fixed_batch_epsilon.py
"""
import json
import math
import os

from dp_accounting import dp_event, rdp

DELTA = 1e-5
ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
SMALLEST_CLIENT = {"ulb": 940, "ieee": 13_559, "elliptic": 441}


def epsilon(q, sigma, steps):
    acc = rdp.RdpAccountant(ORDERS)
    acc.compose(dp_event.PoissonSampledDpEvent(
        q, dp_event.GaussianDpEvent(sigma)), steps)
    return acc.get_epsilon(DELTA)


def main():
    # fixed-batch recipe
    B, sigma_fb, T_fb, E_fb = 128, 1.2, 200, 3
    # uniform-q design
    q_u, sigma_u, T_u, E_u = 0.05, 4.5, 200, 2
    steps_u = T_u * E_u * math.ceil(1 / q_u)
    eps_u = epsilon(q_u, sigma_u, steps_u)

    out = {}
    for name, n in SMALLEST_CLIENT.items():
        q_k = B / n
        steps_k = T_fb * E_fb * math.ceil(n / B)
        out[name] = {"smallest_client": n,
                     "fixed_batch": {"q": q_k, "steps": steps_k,
                                     "epsilon": epsilon(q_k, sigma_fb, steps_k)},
                     "uniform_q": {"q": q_u, "steps": steps_u,
                                   "epsilon": eps_u}}
        print(f"{name:9s} n={n:6d}  fixed-batch eps={out[name]['fixed_batch']['epsilon']:7.1f}"
              f"   uniform-q eps={eps_u:.4f}")
    os.makedirs("results", exist_ok=True)
    with open("results/fixed_batch_epsilon.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("saved -> results/fixed_batch_epsilon.json")


if __name__ == "__main__":
    main()
