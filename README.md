# nano-pupil 🎓⚡

> **Distilling time-series foundation models into calibrated microcontrollers (<256 KB).**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2610.XXXXX-B31B1B.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the paper:  
**"Does a Distilled Student Inherit Its Teacher’s Calibration? Quantile Distillation of Time-Series Foundation Models for Microcontroller Unit Deployments"**

---

## 📌 Abstract & Key Findings

While knowledge distillation enables real-time point forecasting on edge devices, it remains unclear whether a ultra-compact student model retains the **full predictive distribution and calibration** of a massive time-series foundation model (e.g., TimesFM, Chronos).

`nano-pupil` investigates distribution distillation for sub-256 KB models targeted for microcontrollers, featuring:

1. **Full Distribution Distillation:** Direct alignment of target quantile predictions using multi-quantile pinball and Continuous Ranked Probability Score (CRPS) teacher-student loss.
2. **Calibration Transfer Analysis:** Empirical bounds on when sub-256 KB quantized students maintain operational prediction intervals (coverage calibration) vs. when they experience severe calibration drift under low model capacity.
3. **Benchmarking vs. Scratch Training:** Direct comparisons against custom INT8 models trained from scratch (e.g., TinyCast).

---

## 🛠 Repository Structure

```text
nano-pupil/
├── data/              # Benchmarks & preprocessing scripts (ETTh, Electricity, Synthetic)
├── models/
│   ├── teacher/       # Foundation model wrappers (TimesFM, Chronos)
│   └── student/       # Sub-256 KB backbone architectures (MLP-Mixer, TCN)
├── losses/            # Distribution distillation losses (CRPS, Quantile Pinball)
├── quant/             # Post-Training Quantization (PTQ) & Quantization-Aware Training (QAT)
├── scripts/           # Training, evaluation, and C-code export routines
├── eval_calibration.py # Coverage, sharpness, and CRPS evaluation pipeline
└── train_distill.py   # Primary distillation runner
