# Reproduction guide

## Environment and workflows

Install [requirements.txt](../requirements.txt) from the root. These packages cover the `project/` experiment; legacy `mbt_gym/` integrations may need additional packages.

Set the BTC CSV path without editing source:

```powershell
$env:BTC_DATA_PATH = "C:/data/btc_usdt_perp_1m.csv"
Set-Location project
python train_data.py
python make_figures.py
```

For synthetic experiments use `python train_ppo.py` from `project/`. The empirical and synthetic workflows represent different data-generating processes.

## Inputs and outputs

The empirical workflow needs chronological 1-minute BTC OHLCV data. Match timestamp parsing and required columns to `project/train_data.py` and `project/envs/lob_data_env.py`. Training produces local `project/outputs/`; selected figures and model artifacts are committed under `project/results/`. The figure script may need data and checkpoints beyond the committed figures.

## Evaluation chronology

The configured training period is June 2020–September 2022, while `COVID_Crash_2020Q1` covers March–April 2020. That regime is a retrospective stress test using training data from later dates, not a chronological forward test. Later regimes should also be checked for calibration and selection leakage before interpreting performance.

This cleanup retains the experiment design and previously reported numbers. Record seeds, checkpoints, normalization, fill rules and costs for a reproducible comparison.

## Run record

For every result you intend to compare or share, retain the repository commit, exact command/configuration, input-data identity, environment versions and output location. Save a dependency snapshot after a successful reference run; newly added requirements lists are intentionally unpinned until that validation exists.

## Validation status

Cleanup review: 14 October 2026. File paths and documented entry points were checked against the checkout. Full empirical results were not regenerated. Project-specific external prerequisites and known blockers are listed above.
