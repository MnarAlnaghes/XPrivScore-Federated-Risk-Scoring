# X-PrivScore

Code and raw results for the paper **"Collusion-Resilient Federated Risk
Scoring under Uniform Record-Level Differential Privacy"** (M. Alnaghes).

X-PrivScore combines federated training with uniform-q record-level
DP-SGD (Opacus), Adaptive Mask Rotation (AMR) secure aggregation, and
coordinate-wise trimmed-mean aggregation for Byzantine robustness.

## Layout

```
code/                 training pipeline, AMR, accountant, benchmarks
  run_experiment.py   entry point for every training experiment
  train_federated.py  federated loop with uniform-q DP-SGD
  amr.py              Adaptive Mask Rotation + pairwise-masking baseline
  aggregation.py      trimmed-mean / FedAvg aggregation
  privacy_and_explain.py  membership inference and DP-SHAP explanations
  amr_microbench.py   AMR correctness, collusion Monte-Carlo, latency (N=10)
  fixed_batch_epsilon.py  Table 5 accountant comparison
  amr_scaling_study.py   AMR collusion/latency/correctness vs N
  tests/smoke_test.py
results/              one JSON per experimental condition (see table below)
  superseded/         an earlier run kept for transparency (see note)
scripts/reproduce_all.sh   exact command for every file in results/
notebooks/            Colab runners used for the GPU experiments
data/README.md        where to place the three public datasets
```

## Setup

```bash
pip install -r requirements.txt
# place the datasets under data/ as described in data/README.md
python code/tests/smoke_test.py          # quick sanity check
```

Run all commands from the repository root.

## Where each number in the paper comes from

| Paper item | Result file(s) |
|---|---|
| epsilon = 4.7849 (uniform-q, all clients) | `python code/run_experiment.py --plan --dataset ulb --sigma 4.5` |
| Table 3 (AMR vs pairwise latency, d = 4k / 18k) | `results/amr_micro.json` |
| Sec. V-A / IX-J mask-cancellation error; Sec. V-B / IX-J collusion 1.44% and 1.58% | `results/amr_micro.json` |
| Table 5 (fixed-batch vs uniform-q worst-case epsilon) | `python code/fixed_batch_epsilon.py` -> `results/fixed_batch_epsilon.json` |
| Table 6 (main results) | `t3_ulb_mlp`, `t3_ieee_mlp_v2`, `t3_elliptic_mlp` |
| Table 7 (architecture study) | `t3_elliptic_mlp`, `t3_elliptic_tt` |
| Sec. IX-C (ten-seed confirmation) | `main10_ulb`, `main10_ieee`, `main10_elliptic` |
| Sec. IX-K worst-case epsilon at large K | `epsilon_worst_case` field of `kscale_*` |
| Table 8 (privacy-utility) | `t4_sigma6`, `t3_ulb_mlp`, `t4_sigma3`, `t4_nodp` |
| Table 9 (attacks), b = 1 | `t5b_{label_flip,scaling,collusion}[_flonly]` |
| Table 9 (attacks), b = 2 | `t5_{label_flip,scaling,collusion}[_flonly]` |
| Table 10 (component ablation) | `t8_central`, `t8_flmpc`, `t8_flonly`, `t8_fldp` |
| Table 11 (scalability in K) | `kscale_{ulb,ieee,elliptic}_K{20,50,100}` |
| Table 12; Sec. IX-K collusion and correctness across N | `results/amr_scaling.json` |
| Table 13 (tightening k_trim) | `ktrim{2,3}_benign`, `ktrim{2,3}_scaling_b2` |

Every training result file stores its configuration in `summary.config`
plus per-seed metrics in `per_seed`. `scripts/reproduce_all.sh` lists the
exact command that produced each file.

## Reproducibility notes

* Seeds are `{0,1,2}` (ten-seed runs: `0..9`). Training results reproduce
  up to GPU floating-point variation; experiments were run on one NVIDIA T4.
* AMR collusion and correctness results are deterministic CPU computations.
  Latency is wall-clock time and depends on the machine: the AMR/pairwise
  ratio reproduces (about 0.2), the absolute milliseconds do not.
* `results/superseded/t3_ieee_mlp.json` is an earlier IEEE-CIS run made
  before the feature-selection fix in `code/data.py` (that run selected
  scale-dominated columns such as the raw timestamp). The paper reports
  `t3_ieee_mlp_v2.json`, produced with the current code.

## Data

All three datasets are public; none are redistributed here.
See `data/README.md`.

## License

MIT (see `LICENSE`).
