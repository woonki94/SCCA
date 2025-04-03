# Efficient Stochastic Optimization for Canonical Correlation Analysis (CCA)

This repository contains an implementation of the algorithm described in the paper:

> **Efficient Globally Convergent Stochastic Optimization for Canonical Correlation Analysis**  
> Weiran Wang, Jialei Wang, Raman Arora  
> [arXiv:1604.01870](https://arxiv.org/abs/1604.01870)

## Overview

Canonical Correlation Analysis (CCA) is a classical technique for identifying relationships between two multivariate datasets. This implementation follows the stochastic optimization framework presented in the paper, featuring:

- **SVRG-based stochastic gradient optimization** with global convergence guarantees.
- **Shift-and-Invert (SI) preconditioning** combined with SVRG for improved numerical stability.
- Experiments on both **synthetic data** and real-world **Fashion-MNIST** to evaluate performance in practice.

## Algorithm Variants Implemented

This repo includes two main algorithmic variants:

### 1. **SVRG for CCA**
- Reduces gradient variance using full gradient snapshots.
- Improves convergence while keeping the algorithm scalable and stochastic.

### 2. **Shift-and-Invert (SI) with SVRG**
- Improves conditioning of the problem by solving a shifted and inverted system.
- Further accelerates convergence in challenging settings (e.g. poorly conditioned covariance matrices).

##  Practical Experiment
- The standard Fashion-MNIST dataset is split into two views (e.g., left/right pixels, odd/even pixels).
- Demonstrates how CCA embeddings can uncover correlated structure between real-world views.

### Synthetic Experiment
- Randomly generated paired views `(X, Y)` with controlled correlation.
- Useful for verifying convergence behavior and canonical correlation recovery.

## Report

You can view the detailed project report here:  
[`/report/AI586_Final_Project_Woonki_Kim`](./report/AI586_Final_Project_Woonki_Kim.pdf)


@article{wang2016efficient,
  title={Efficient globally convergent stochastic optimization for canonical correlation analysis},
  author={Wang, Weiran and Wang, Jialei and Arora, Raman},
  journal={arXiv preprint arXiv:1604.01870},
  year={2016}
}
