#!/usr/bin/env bash
# Regenerates every file in results/ with the exact configuration behind it.
# Run from the repository root after placing the datasets under data/
# (see data/README.md). Each training run averages the seeds given; on a
# single NVIDIA T4 a 3-seed run takes roughly 1-3 hours depending on the
# dataset, so run the sections you need.
set -euo pipefail
R="python code/run_experiment.py"
SEEDS10="0 1 2 3 4 5 6 7 8 9"

# --- Privacy accountant only (no training): epsilon = 4.7849 at delta=1e-5
$R --plan --dataset ulb --sigma 4.5

# --- Main results (Table 6) and architecture study (Table 7)
$R --dataset ulb      --model mlp            --name t3_ulb_mlp
$R --dataset ieee     --model mlp            --name t3_ieee_mlp_v2
$R --dataset elliptic --model mlp            --name t3_elliptic_mlp
$R --dataset elliptic --model tabtransformer --name t3_elliptic_tt

# --- Ten-seed confirmation (Section IX-C)
$R --dataset ulb      --model mlp --seeds $SEEDS10 --name main10_ulb
$R --dataset ieee     --model mlp --seeds $SEEDS10 --name main10_ieee
$R --dataset elliptic --model mlp --seeds $SEEDS10 --name main10_elliptic

# --- Privacy-utility trade-off (Table 8); sigma=4.5 row is t3_ulb_mlp
$R --dataset ulb --model mlp --sigma 3.0 --name t4_sigma3
$R --dataset ulb --model mlp --sigma 6.0 --name t4_sigma6
$R --dataset ulb --model mlp --no-dp     --name t4_nodp

# --- Robustness under attack (Table 9): b=1 -> t5b_*, b=2 -> t5_*
for ATT in label_flip scaling collusion; do
  $R --dataset ulb --model mlp --attack $ATT --n-malicious 1 --name t5b_$ATT
  $R --dataset ulb --model mlp --attack $ATT --n-malicious 1 \
     --no-dp --no-mpc --aggregation fedavg --name t5b_${ATT}_flonly
  $R --dataset ulb --model mlp --attack $ATT --n-malicious 2 --name t5_$ATT
  $R --dataset ulb --model mlp --attack $ATT --n-malicious 2 \
     --no-dp --no-mpc --aggregation fedavg --name t5_${ATT}_flonly
done

# --- Component ablation (Table 10)
$R --dataset ulb --model mlp --centralized --no-dp            --name t8_central
$R --dataset ulb --model mlp --no-dp                          --name t8_flmpc
$R --dataset ulb --model mlp --no-dp --no-mpc --aggregation fedavg --name t8_flonly
$R --dataset ulb --model mlp                                  --name t8_fldp

# --- Scalability across federation size (Table 11)
for K in 20 50; do
  $R --dataset ulb --model mlp --n-clients $K --name kscale_ulb_K$K
done
for K in 20 50 100; do
  $R --dataset ieee     --model mlp --n-clients $K --name kscale_ieee_K$K
  $R --dataset elliptic --model mlp --n-clients $K --name kscale_elliptic_K$K
done

# --- Robustness envelope: tightening k_trim (Table 13)
for KT in 2 3; do
  $R --dataset ulb --model mlp --k-trim $KT --name ktrim${KT}_benign
  $R --dataset ulb --model mlp --k-trim $KT --attack scaling --n-malicious 2 \
     --name ktrim${KT}_scaling_b2
done

# --- AMR micro-benchmarks and scaling study (CPU only, minutes)
python code/amr_microbench.py
python code/amr_scaling_study.py
