# PDoS: A Profitable Denial-of-Service Attack against Proof-of-Work Blockchain Liveness

This repository contains the official, fully anonymized artifacts (source code, datasets, and evaluation scripts) for our ACM CCS 2026 double-blind submission.

## 📂 Repository Structure

- `config.py`: Core configurations.
- `theory.py`: Mathematical models and CTMC calculator.
- `simulator.py`: Monte Carlo simulation engine.
- `data.py`: Data preprocessing and cleaning scripts.
- `dataset/`: Preprocessed Bitcoin data and BigQuery extraction scripts (`bitcoin_raw.sql`).
- `precompute/`: Scripts to generate computationally intensive caches.
- `plotter/`: Scripts to reproduce all figures presented in the paper.
- `print_tables.py`: Script to generate Tables.

> **Note:** The `cache/` and `figures/` directories are git-ignored and will be automatically generated during the reproduction process.

## ⚙️ Installation

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

## 🚀 Reproduction Guide

To fully reproduce the results, please execute the following steps sequentially.

### Step 1: Precomputation (Caching)
Due to high computational complexity, you **MUST** build the local caches for CTMC grid search and trace-driven Nash equilibrium replay first:

```bash
python precompute/precompute_threshold_theory.py
python precompute/precompute_threshold_sim.py
python precompute/precompute_net_cost.py
python precompute/precompute_trace_nash.py
python precompute/precompute_trace_replay.py
```

### Step 2: Generate Tables
Output Dataset Statistics and Parameter Estimation directly to the console:

```bash
python print_tables.py
```

### Step 3: Render Figures
Instantly generate the high-quality vector graphics (PDFs) used in the paper. Outputs will be saved to the `figures/` directory:

```bash
python plotter/plot_fig1.py
python plotter/plot_fig2.py
python plotter/plot_fig3.py
python plotter/plot_fig4.py
python plotter/plot_fig5.py
```

## 🛡️ Open Science & Ethical Considerations
This research involves no ethical risks. We fully endorse the Open Science Policy. The formal proofs for all theorems are provided in the paper's Appendix, and this artifact repository is open-sourced to grant the scientific community unrestricted rights to review, validate, and build upon our work.
