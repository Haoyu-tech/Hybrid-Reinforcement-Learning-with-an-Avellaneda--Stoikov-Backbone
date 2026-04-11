# Hybrid-Reinforcement-Learning-with-an-Avellaneda--Stoikov-Backbone

This folder contains the runnable code for Chapter 2 of the market-making project.  
The current focus is **market making with Avellaneda-Stoikov (AS), Pure RL, and Hybrid AS+RL**, with the **BTC 1-minute empirical experiment** as the main result line.

## What This Project Does

There are two experiment lines in this folder:

- `train_data.py`
  Runs the **real-data empirical study** on Binance BTC/USDT perpetual 1-minute OHLCV data.
  It calibrates an AS benchmark from the training sample, trains `Pure RL` and `Hybrid AS+RL`, and evaluates all three methods on six out-of-sample BTC regimes.

- `train_ppo.py`
  Runs the **synthetic simulation study** in a stylized LOB environment.
  This is useful for controlled methodology checks, but it is not the main empirical result line.

- `make_figures.py`
  Collects experiment outputs and generates paper-style figures and tables in `project/results/`.

## Current Research Focus

The current main pipeline is:

1. Use real BTC 1-minute data.
2. Build an OHLC replay market-making environment.
3. Use a calibrated `Static AS` model as the benchmark.
4. Train and compare:
   - `Static AS`
   - `Pure RL`
   - `Hybrid AS+RL`
5. Export paper-ready figures and tables.

## Directory Structure

```text
project/
├─ agents/
│  └─ ppo.py                  # PPO policy and trainer
├─ envs/
│  ├─ lob_env.py              # synthetic LOB environment
│  └─ lob_data_env.py         # BTC minute-bar replay environment
├─ models/
│  └─ avellaneda_stoikov.py   # AS benchmark / backbone
├─ utils/
├─ outputs/                   # training checkpoints and raw experiment outputs
├─ results/                   # paper figures and summary tables
├─ train_data.py              # BTC empirical experiment
├─ train_ppo.py               # synthetic simulation experiment
└─ make_figures.py            # figure/table generation
```

## Environment

Recommended Python version:

- `Python 3.10+`

Main dependencies:

- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `tqdm`
- `Pillow`

If you want to generate PowerPoint materials outside this folder, `python-pptx` may also be useful, but it is not required for the core project pipeline.

## Real BTC Empirical Pipeline

### 1. Set the BTC data path

In `train_data.py`, the real data CSV path is defined by:

```python
DATA_PATH = "E:/CTA分钟级别/data/binance_btc_perp/btc_usdt_perp_1m_from_2020.csv"
```

Before running the empirical experiment, update this path if your data is stored elsewhere.

Expected columns include at least:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

### 2. Run the empirical experiment

From the `project/` directory:

```bash
python train_data.py
```

What this script does:

- loads BTC 1-minute OHLCV data
- splits training and out-of-sample test periods
- estimates volatility and regime thresholds
- calibrates the AS benchmark from BTC training data
- trains `Pure RL` and `Hybrid AS+RL`
- evaluates `Static AS`, `Pure RL`, and `Hybrid AS+RL`
- saves raw outputs, models, figures, and paper tables

### 3. Generate figures and paper tables

```bash
python make_figures.py
```

This script reads `project/outputs/` and writes formatted results to `project/results/`.

## Synthetic Pipeline

To run the simulation baseline:

```bash
python train_ppo.py
```

This is useful for:

- debugging training logic
- checking whether the AS prior behaves correctly in a controlled setting
- comparing synthetic and empirical conclusions

## Empirical Setup Summary

The BTC empirical environment currently uses:

- episode length: `500` one-minute bars
- within-episode price normalization: `S0 = 100`
- tick size: `0.01`
- max inventory: `10`
- maker fee rate: `2e-4`
- inventory penalty: `0.005`
- adverse-selection penalty: `0.3`
- terminal penalty: `0.10`
- flatten fee rate: `7e-4`

The AS benchmark is calibrated from BTC training data using:

- fill-curve fitting for `k` and `A`
- grid search over `gamma` and `horizon`

## Main Outputs

### Raw outputs

Generated in `project/outputs/`:

- `data_pure_rl_final.pt`
- `data_hybrid_final.pt`
- `data_period_results.csv`
- `paper_table_overall.csv`
- `paper_table_by_period.csv`
- `as_calibration_summary.csv`
- training curve PNGs

### Paper-style outputs

Generated in `project/results/`:

- `fig1_pnl_comparison.pdf`
- `fig2_risk_metrics.pdf`
- `fig3_behavior.pdf`
- `fig5_fill_curve_calibration.pdf`
- `fig8_sharpe_maxdd_scatter.pdf`
- `fig11_btc_equity_curves.pdf`
- `fig12_btc_terminal_distribution.pdf`
- summary tables in CSV and LaTeX format

## Interpretation of the Two RL Models

- `Pure RL`
  A model-free PPO policy that directly controls quote offsets and sizes.

- `Hybrid AS+RL`
  A structured PPO policy that starts from the AS backbone and learns residual corrections.

- `Static AS`
  A calibrated benchmark used for comparison in both the empirical study and methodology discussion.

## Notes and Caveats

- The empirical environment uses **minute-bar OHLC replay**, not full L2 order book replay.
- Fill logic is based on **OHLC crossing**, not true queue-level execution.
- Price is normalized to `S0 = 100` within each episode to make the policy scale-stable across the BTC price history.
- If `torch` raises Windows DLL initialization errors, keep the same environment-variable pattern used in the current scripts.

## Recommended Reproduction Order

If you want the main Chapter 2 result with minimum confusion, use this order:

1. `train_data.py`
2. `make_figures.py`

Only run `train_ppo.py` if you also want the synthetic comparison line.

## Contact Between Code and Paper Assets

This `project/` folder produces the numerical results and figures.  
The paper assembly assets live outside this folder in:

- `../4.11Chapter2_work/`

That folder contains the LaTeX manuscript, selected figures, and packaged empirical paper materials.
