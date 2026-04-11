# Hybrid-Reinforcement-Learning-with-an-Avellaneda--Stoikov-Backbone

This folder contains the runnable code for Chapter of the market-making.  
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

## Visual Overview

### Out-of-sample PnL across six BTC regimes

![BTC regime PnL comparison](project/results/fig1_pnl_comparison.png)

### BTC empirical equity curves across all six test periods

![BTC empirical equity curves](project/results/fig11_btc_equity_curves.png)

### Terminal PnL distributions by BTC regime

![BTC terminal PnL distributions](project/results/fig12_btc_terminal_distribution.png)

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

### 1.1 Recommended minute-level data format

The empirical pipeline assumes a **1-minute OHLCV CSV**.

Minimum requirements:

- one row = one minute
- rows sorted in chronological order
- `timestamp` in milliseconds or convertible to pandas datetime
- no missing values in `open/high/low/close/volume`

Recommended checks before running:

```python
import pandas as pd

df = pd.read_csv("your_btc_1m.csv")
print(df.head())
print(df.columns)
print(df.isna().sum())
```

If your data uses different column names, rename them before running the script.

Example:

```python
df = df.rename(columns={
    "ts": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
})
```

### 1.2 What “testing with minute-level data” means here

This project does **not** replay full L2 order book messages.
Instead, it tests market-making policies in a **minute-bar replay environment** built from BTC 1-minute OHLCV data.

The logic is:

1. Use real BTC 1-minute historical bars.
2. Build episodes of length `500` minutes.
3. Let each agent post bid/ask quotes at every minute.
4. Approximate fills using the minute bar:
   - bid fills if `Low_t <= bid_price`
   - ask fills if `High_t >= ask_price`
5. Update cash, inventory, fees, adverse-selection cost, and terminal liquidation cost.
6. Compute out-of-sample metrics such as:
   - net PnL
   - Sharpe
   - Sortino
   - max drawdown
   - average inventory
   - fill rates

So “using minute-level data to test” in this project means:

- **real BTC price path**
- **OHLC-based fill approximation**
- **episode-based out-of-sample evaluation**

not full exchange message replay.

### 1.3 Visual intuition for the minute-level test

The following figures are especially useful when explaining what the minute-level empirical test is doing:

- `results/fig5_fill_curve_calibration.png`
  shows how the AS benchmark is calibrated from BTC minute data
- `results/fig11_btc_equity_curves.png`
  shows cumulative episode-level PnL paths across six BTC regimes
- `results/fig12_btc_terminal_distribution.png`
  shows whether the performance improvement is distribution-wide rather than driven by a few episodes

#### AS calibration from minute-bar fill behavior

![AS fill-curve calibration](results/fig5_fill_curve_calibration.png)

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

### 2.1 Detailed minute-level testing workflow

If you want to understand exactly how the minute-level empirical test works, the pipeline is:

#### Step A. Load the full BTC minute dataset

`train_data.py` reads the CSV and converts `timestamp` into a datetime column:

- source file: `train_data.py`
- loaded into `df_full`

This is the full historical BTC 1-minute dataset used for both training and out-of-sample testing.

#### Step B. Split training and test periods

The script uses:

- training window: `2020-06-01 ~ 2022-09-30`
- six out-of-sample BTC test periods:
  - `COVID_Crash_2020Q1`
  - `FTX_Collapse_2022Q4`
  - `Bull_Run_2023Q1`
  - `LowVol_Sideways_2023Q3`
  - `ETF_Rally_2024Q1`
  - `ATH_2024Q4`

This matters because the test is **strictly out-of-sample**.
The model is not evaluated on the same minutes it was trained on.

#### Step C. Estimate volatility and market regimes from the training set

Before testing, the code computes:

- minute-level rolling volatility
- episode-level effective volatility `sigma_eff`
- low / medium / high volatility thresholds

These values are used by:

- the AS benchmark
- the hybrid backbone
- the observation features inside the empirical environment

#### Step D. Calibrate the AS benchmark from BTC training data

The benchmark is not a naive default AS model.
It is first calibrated on BTC minute data by:

1. posting symmetric quotes at different offsets
2. measuring empirical fill probability from OHLC crossing
3. fitting the AS fill curve parameters `k` and `A`
4. grid searching `gamma` and `horizon`

This step is important because otherwise `Static AS` can become unrealistically passive and almost never trade.

#### Step E. Construct the minute-bar replay environment

The empirical environment is implemented in:

- `envs/lob_data_env.py`

At each minute:

- the current bar provides `open/high/low/close/volume`
- the agent submits `[log_bid_offset, log_ask_offset, log_bid_size, log_ask_size]`
- prices are normalized within each episode to `S0 = 100`
- fills are approximated from bar `high/low`

Inventory and PnL are then updated using:

- maker fees
- cancellation cost
- inventory penalty
- adverse-selection penalty
- terminal flattening cost

#### Step F. Train RL models on the training window

Two learned policies are trained:

- `Pure RL`
- `Hybrid AS+RL`

The benchmark:

- `Static AS`

is not trained with PPO.
It is directly evaluated after calibration.

#### Step G. Evaluate on each minute-level out-of-sample period

For each test period:

1. the code samples multiple episode start points
2. each episode runs for `500` minutes
3. each of the three agents is tested on the same period
4. metrics are averaged across episodes

This is the main “minute-level data test”.

#### Step H. Save result files

The empirical test writes:

- `outputs/data_period_results.csv`
- `outputs/paper_table_overall.csv`
- `outputs/paper_table_by_period.csv`
- trained model checkpoints
- training curves
- comparison figures

### 2.5 What the minute-level test results look like

#### Regime-level performance comparison

![Regime-level BTC performance](results/fig2_risk_metrics.png)

#### Execution and inventory behavior

![BTC execution and inventory behavior](results/fig3_behavior.png)

### 2.2 How to test only with existing trained models

If you already have:

- `data_pure_rl_final.pt`
- `data_hybrid_final.pt`
- `as_calibration_summary.csv`

you can reuse the saved outputs to generate figures without retraining:

```bash
python make_figures.py
```

This does not retrain PPO.
It reads existing result files and generates paper-ready plots and tables.

### 2.3 How to change the minute-level test setup

The most common changes are:

- change the BTC file path
- change the train/test dates
- change the episode length
- change transaction cost assumptions
- change the number of test episodes

Relevant locations:

- `DATA_PATH` in `train_data.py`
- `TRAIN_PERIOD` in `train_data.py`
- `TEST_PERIODS` in `train_data.py`
- `ENV_KWARGS["episode_steps"]` in `train_data.py`
- `evaluate_period(..., n_episodes=20)` in `train_data.py`

### 2.4 Common mistakes when testing with minute-level data

- Using unsorted rows.
  The environment assumes time is increasing.

- Using missing or duplicated minutes.
  This can distort volatility estimation and episode construction.

- Mixing training and testing windows.
  The empirical conclusions are intended to be out-of-sample.

- Interpreting OHLC replay as true order book replay.
  This project uses a practical approximation, not true queue-level fills.

- Forgetting to recalibrate AS when changing the dataset.
  The benchmark should be fitted to the new minute-level sample.

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
