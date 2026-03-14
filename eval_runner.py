from __future__ import annotations

import argparse
import os
import pickle
import sys

# Force CPU-only path for robust torch loading on this machine.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO

from config import CFG
from evaluate import evaluate_across_regimes, to_comprehensive_table_df, to_detailed_table_df, to_regime_summary_df, to_table_df

OUT = "outputs"
RESULTS = os.path.join(OUT, "results")
MODELS = os.path.join(OUT, "models")
os.makedirs(RESULTS, exist_ok=True)


def _try_load_all(prefix: str):
    models = []
    for i in range(CFG.eval.n_seeds):
        seed = CFG.train.seed + i * CFG.eval.train_seed_stride
        path = os.path.join(MODELS, f"ppo_{prefix}_seed{seed}_final.zip")
        if not os.path.exists(path):
            continue
        try:
            print(f"Loading model: {path}")
            models.append(PPO.load(path))
        except Exception as e:
            print(f"[WARN] Skip loading {path}: {e}")
    return models


def run(episodes: int, regimes=None):
    regime_list = list(regimes or CFG.eval.regimes)
    primary_regime = regime_list[0]
    hybrid_model = _try_load_all(f"hybrid_{primary_regime}")
    model_free_model = _try_load_all(f"model_free_{primary_regime}")
    hybrid_tuning_model = _try_load_all(f"hybrid_tuning_{primary_regime}")

    regime_results = evaluate_across_regimes(
        hybrid_model=hybrid_model,
        model_free_model=model_free_model,
        hybrid_tuning_model=hybrid_tuning_model,
        cfg=CFG,
        n_episodes=episodes,
        regimes=regime_list,
    )
    results = regime_results[primary_regime]

    df = to_table_df(results)
    detailed_df = to_detailed_table_df(results)
    comprehensive_df = to_comprehensive_table_df(results)
    regime_df = to_regime_summary_df(regime_results)

    print("\n" + df.to_string(index=False))

    with open(os.path.join(RESULTS, "results.pkl"), "wb") as f:
        pickle.dump(regime_results, f)
    df.to_csv(os.path.join(RESULTS, "table_results.csv"), index=False)
    detailed_df.to_csv(os.path.join(RESULTS, "table_results_detailed.csv"), index=False)
    comprehensive_df.to_csv(os.path.join(RESULTS, "table_results_comprehensive.csv"), index=False)
    regime_df.to_csv(os.path.join(RESULTS, "table_results_by_regime.csv"), index=False)
    print(f"\nSaved evaluation to: {RESULTS}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--regimes", nargs="+", default=list(CFG.eval.regimes))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.episodes, regimes=args.regimes)
