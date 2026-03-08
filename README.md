# SpectralKrum
**A Spectral–Geometric Defense Against Byzantine Attacks in Federated Learning**

This repository contains the reference implementation of **SpectralKrum**, a Byzantine-robust aggregation rule for Federated Learning (FL) that combines **spectral subspace analysis** with **geometric neighbor-based selection** and a **soft orthogonal-energy guard**.

Federated Learning enables decentralized model training without sharing raw data, but it is vulnerable to **Byzantine clients** that submit arbitrarily corrupted updates. Classical robust aggregation methods such as coordinate-wise median, Trimmed Mean, Krum, and Bulyan rely on strong assumptions about gradient clustering that often fail under **non-IID client data** and **adaptive adversaries**.

SpectralKrum addresses this challenge by exploiting the **low-dimensional structure of benign optimization trajectories** observed across FL rounds.

---

## Key Idea

SpectralKrum integrates **spectral filtering** and **geometric robustness** in a single aggregation pipeline:

1. **Historical Subspace Estimation**
   A rolling buffer of past aggregated updates is maintained. PCA is applied (with trimming) to estimate a low-dimensional subspace that captures benign optimization dynamics while reducing contamination from adversarial updates.

2. **Spectral Projection**
   Incoming client updates are projected into the learned subspace, where benign updates cluster more tightly despite non-IID data.

3. **Geometric Selection (Krum in Spectral Space)**
   Krum selection is performed in the reduced-dimensional spectral space, improving robustness when benign updates are dispersed in the original high-dimensional parameter space.

4. **Soft Orthogonal-Energy Guard**
   Selected updates are reweighted using bounded soft weights calibrated from a blend of historical and current-round residual statistics. An adaptive threshold (median + 3·MAD, blended with buffer history) controls the decay, while a configurable safety cap hard-rejects only the worst outliers. This replaces the original hard binary cutoff, reducing over-filtering of honest clients under non-IID heterogeneity.

This design preserves FL's privacy guarantees: **no client data, labels, or trusted server dataset is required**.

---

## What This Repository Contains

- **SpectralKrum aggregation rule** (PCA + Krum + soft orthogonal-energy guard)
- Implementations of **baseline robust aggregators**:
  - Trimmed Mean
  - Coordinate Median
  - Geometric Median
  - Krum / Multi-Krum
  - Bulyan
  - DnC-PMF and DnC-Cluster
- Implementations of **Byzantine attacks**, including:
  - Sign-flip
  - Label-flip (auto-resolved per dataset)
  - Min-max
  - Buffer-drift (subspace manipulation)
  - Adaptive-steer (subspace-aware attack)
  - Semantic backdoor
- Experimental pipeline for **non-IID Federated Learning** using Dirichlet data partitions
- Logging, evaluation, and plotting utilities for:
  - Per-round accuracy
  - AUC (mean accuracy across rounds)
  - Computational overhead comparison (wall-clock breakdown with LaTeX export)

---

## Experimental Setting

- **Datasets**: CIFAR-10, CIFAR-100, MNIST, FEMNIST (EMNIST ByClass)
- **Models**: TinyCNN, MobileNetV2 (adapted for small images)
- **Clients**: 50 total per round
- **Non-IID partitioning**: Dirichlet distribution (α ∈ {0.1, 0.5, 1.0})
- **Byzantine clients**: up to 20% per round
- **Seeds**: 5 independent runs per configuration
- **Evaluation**: accuracy, ASR, guard precision metrics across all configurations

---

## Key Findings

- SpectralKrum is **competitive under directional and subspace-aware attacks** (e.g., adaptive-steer, buffer-drift).
- It **does not dominate** simpler statistical defenses under attacks that remain spectrally indistinguishable from benign updates (e.g., label-flip, min-max).
- The soft guard with adaptive tau **reduces false rejection of honest clients** under high non-IID heterogeneity compared to the original hard cutoff.
- The results highlight **when spectral geometry helps** and **where it fundamentally fails**, emphasizing the need for hybrid defenses.

---

## Scope and Limitations

SpectralKrum is **not a universal Byzantine defense**. It is designed to study and exploit spectral structure in FL optimization and to expose the limits of spectral filtering under adaptive adversaries. The repository is intended for **research and experimental analysis**, not as a drop-in production system.

---

## Citation

If you use this code, please cite:

> **SpectralKrum: A Spectral-Geometric Defense Against Byzantine Attacks in Federated Learning**
> Aditya Tripathi, Karan Sharma, Rahul Mishra, Tapas Kumar Maiti
