# X-PrivScore
### Collusion-Resilient Federated Risk Scoring via Adaptive Mask Rotation and Explainable Transformers

X-PrivScore is a privacy-preserving federated learning framework designed for fraud detection and transaction risk scoring in distributed financial environments.

The framework combines:

- Federated Learning (FL)
- Differential Privacy (DP)
- Adaptive Mask Rotation (AMR)
- Tab-Transformer architectures
- Explainable AI (SHAP)

to provide secure, interpretable, and collusion-resilient risk scoring across decentralized clients without sharing raw transactional data.

---

# Key Features

## Privacy-Preserving Federated Learning
Train collaborative fraud detection models without exposing raw client data.

## Adaptive Mask Rotation (AMR)
A novel secure aggregation mechanism designed to provide cross-round collusion resilience in federated environments.

## Transformer-Based Risk Scoring
Uses Tab-Transformer architectures to capture complex inter-feature dependencies in high-dimensional transaction data.

## Differential Privacy Integration
Supports DP-compatible training using gradient clipping and Gaussian noise injection.

## Explainable AI Layer
Integrates SHAP-based explainability for transparent and interpretable transaction risk analysis.

## Multi-Dataset Evaluation
Validated across multiple real-world fraud and blockchain datasets under non-IID federated settings.

---

# Supported Datasets

- UCI Credit Card Fraud Detection
- IEEE-CIS Fraud Detection
- Elliptic Bitcoin Dataset

Datasets are automatically downloaded or loaded locally and are not included in this repository.

---

# Repository Structure

```bash
run_experiments.py           # Main experiment runner
xprivscore_architecture.py  # Transformer + MLP architecture
dataset_pipeline.py         # Dataset loading and federated partitioning
requirements.txt            # Python dependencies
```

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Run Experiments

## UCI Dataset

```bash
python run_experiments.py --dataset uci
```

## IEEE-CIS Dataset

```bash
python run_experiments.py --dataset ieee_cis
```

## Elliptic Dataset

```bash
python run_experiments.py --dataset elliptic
```

## Run All Experiments

```bash
python run_experiments.py --all
```

---

# Research Focus

- Federated Learning
- Privacy-Preserving AI
- Explainable AI
- Fraud Detection
- AI Security
- Distributed Risk Scoring

---

# License

MIT License

---

# Author

Dr. Manar Alnaghes  
AI Governance & Privacy-Preserving AI Researcher
