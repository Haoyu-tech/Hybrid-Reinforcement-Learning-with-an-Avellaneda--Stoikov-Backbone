"""Main entry for mbt_gym-compatible training/evaluation pipeline."""
# D:\software\anaconda3\envs\quant\python.exe "MBT-gym files/main.py" --mode full --steps 10000 --env-steps 400 --episodes 50
# D:\software\anaconda3\envs\quant\python.exe "MBT-gym files/main.py" --mode full --steps 10000 --env-steps 400 --episodes 50 --seeds 3
# D:\software\anaconda3\envs\quant\python.exe "MBT-gym files/main.py" --mode full --steps 10000 --env-steps 400 --episodes 50 --seeds 3

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
# Make torch/sb3 import robust on this Windows setup (avoid CUDA DLL init issues).
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import CFG
from mbt_factory import make_trading_env

OUT = "outputs"
MODELS = os.path.join(OUT, "models")
RESULTS = os.path.join(OUT, "results")
FIGURES = os.path.join(OUT, "figures")
for d in [OUT, MODELS, RESULTS, FIGURES]:
    os.makedirs(d, exist_ok=True)


def _training_budget_summary(
    *,
    n_seeds: int,
    total_steps: int | None = None,
    mode: str = "train",
    regimes: list[str] | None = None,
) -> None:
    timesteps = int(total_steps if total_steps is not None else CFG.train.total_timesteps)
    rollout_steps = int(CFG.train.n_steps * CFG.train.n_envs)
    rollout_updates = timesteps / max(rollout_steps, 1)
    optimizer_epochs = rollout_updates * CFG.train.n_epochs
    regime_list = regimes or list(CFG.eval.regimes)

    if mode in {"train", "full"}:
        model_count = 3 * int(n_seeds)
        mode_label = "main training"
    elif mode == "ablation":
        model_count = 7 * int(n_seeds)
        mode_label = "ablation training"
    else:
        return

    total_env_steps = model_count * timesteps

    print("\n[Training Budget]")
    print(f"  Mode              : {mode_label}")
    print(f"  Regimes           : {', '.join(regime_list)}")
    print(f"  Seeds             : {n_seeds}")
    print(f"  Models to train   : {model_count}")
    print(f"  Timesteps/model   : {timesteps:,}")
    print(f"  Rollout size      : {rollout_steps:,} ({CFG.train.n_steps} x {CFG.train.n_envs})")
    print(f"  PPO updates/model : {rollout_updates:.1f}")
    print(f"  Opt epochs/model  : {optimizer_epochs:.1f}")
    print(f"  Total env steps   : {total_env_steps:,}")


def check_install() -> bool:
    ok = True
    for pkg, import_name in [
        ("mbt_gym", "mbt_gym"),
        ("stable_baselines3", "stable_baselines3"),
        ("gymnasium", "gymnasium"),
        ("numpy", "numpy"),
        ("torch", "torch"),
    ]:
        try:
            __import__(import_name)
            print(f"  OK   {pkg}")
        except ImportError:
            print(f"  MISS {pkg}")
            ok = False
    return ok


def run_demo(episodes: int = 3, seed: int = 42) -> None:
    print("\n[DEMO] Running random-policy mbt_gym simulation...")
    env = make_trading_env(cfg=CFG, seed=seed, regime="standard")

    ep_pnls = []
    ep_rewards = []

    for ep in range(episodes):
        _obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0

        wealth0 = float(env.state[0, 0] + env.state[0, 1] * env.state[0, 3])

        while not done:
            action = env.action_space.sample().reshape(1, -1)
            _obs, reward, dones, _info = env.step(action)
            done = bool(np.asarray(dones).reshape(-1)[0])
            total_reward += float(np.asarray(reward).reshape(-1)[0])
            step_count += 1

        s = env.state[0]
        cash, inv, mid = float(s[0]), float(s[1]), float(s[3])
        terminal_pnl = cash + inv * mid - wealth0

        ep_rewards.append(total_reward)
        ep_pnls.append(terminal_pnl)

        print(
            f"  Episode {ep + 1}: steps={step_count}, reward_sum={total_reward:.4f}, "
            f"terminal_pnl={terminal_pnl:.4f}, inv={inv:.2f}"
        )

    env.close()
    print("\n[DEMO] Summary")
    print(f"  Mean terminal PnL: {np.mean(ep_pnls):.4f}")
    print(f"  Std terminal PnL : {np.std(ep_pnls):.4f}")
    print(f"  Mean reward sum  : {np.mean(ep_rewards):.4f}")


def run_training(n_seeds: int = 1, total_steps: int | None = None):
    from train_ppo import train_all

    if total_steps is not None:
        CFG.train.total_timesteps = total_steps
    return train_all(cfg=CFG, n_seeds=n_seeds)


def _try_load_models(prefix: str, n_seeds: int):
    try:
        from stable_baselines3 import PPO
    except Exception:
        return []

    models = []
    for i in range(n_seeds):
        seed = CFG.train.seed + i * CFG.eval.train_seed_stride
        path = os.path.join(MODELS, f"ppo_{prefix}_seed{seed}_final.zip")
        if os.path.exists(path):
            try:
                print(f"Loading model: {path}")
                models.append(PPO.load(path))
            except Exception as e:
                print(f"Skip loading {path}: {e}")
    return models


def run_eval(trained: dict | None = None, n_episodes: int = 20, regimes: list[str] | None = None):
    from evaluate import (
        evaluate_across_regimes,
        to_comprehensive_table_df,
        to_detailed_table_df,
        to_regime_summary_df,
        to_table_df,
    )

    if trained is None:
        trained = {}

    hybrid_model = (trained.get("hybrid", []) if trained else [])
    model_free_model = (trained.get("model_free", []) if trained else [])
    hybrid_tuning_model = (trained.get("hybrid_tuning", []) if trained else [])

    # If eval is called standalone, try loading saved models from disk.
    if not hybrid_model or not model_free_model or not hybrid_tuning_model:
        try:
            from stable_baselines3 import PPO
        except Exception as e:
            PPO = None
            print(f"[WARN] Cannot import stable_baselines3.PPO for model loading: {e}")
            # Fallback: run standalone evaluator in a fresh process.
            runner = os.path.join(THIS_DIR, "eval_runner.py")
            cmd = [sys.executable, runner, "--episodes", str(n_episodes), "--regimes", *(regimes or list(CFG.eval.regimes))]
            print("[INFO] Fallback to subprocess evaluator:", " ".join(cmd))
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
            if proc.returncode != 0:
                raise RuntimeError(f"Subprocess evaluator failed with code {proc.returncode}")
            pkl = os.path.join(RESULTS, "results.pkl")
            if not os.path.exists(pkl):
                raise FileNotFoundError(f"Expected results not found after subprocess eval: {pkl}")
            with open(pkl, "rb") as f:
                return pickle.load(f)

        if not hybrid_model:
            hybrid_model = _try_load_models(f"hybrid_{CFG.eval.regimes[0]}", CFG.eval.n_seeds)
        if not model_free_model:
            model_free_model = _try_load_models(f"model_free_{CFG.eval.regimes[0]}", CFG.eval.n_seeds)
        if not hybrid_tuning_model:
            hybrid_tuning_model = _try_load_models(f"hybrid_tuning_{CFG.eval.regimes[0]}", CFG.eval.n_seeds)

    regime_results = evaluate_across_regimes(
        hybrid_model=hybrid_model,
        model_free_model=model_free_model,
        hybrid_tuning_model=hybrid_tuning_model,
        cfg=CFG,
        n_episodes=n_episodes,
        regimes=regimes,
    )

    primary_regime = (regimes or list(CFG.eval.regimes))[0]
    results = regime_results[primary_regime]
    df = to_table_df(results)
    detailed_df = to_detailed_table_df(results)
    comprehensive_df = to_comprehensive_table_df(results)
    regime_df = to_regime_summary_df(regime_results)
    print(f"\n[Primary regime: {primary_regime}]\n" + df.to_string(index=False))

    with open(os.path.join(RESULTS, "results.pkl"), "wb") as f:
        pickle.dump(regime_results, f)
    df.to_csv(os.path.join(RESULTS, "table_results.csv"), index=False)
    detailed_df.to_csv(os.path.join(RESULTS, "table_results_detailed.csv"), index=False)
    comprehensive_df.to_csv(os.path.join(RESULTS, "table_results_comprehensive.csv"), index=False)
    regime_df.to_csv(os.path.join(RESULTS, "table_results_by_regime.csv"), index=False)
    print(f"\nSaved evaluation to: {RESULTS}")
    return regime_results


def run_ablation_study(n_seeds: int = 1, n_episodes: int = 20, regimes: list[str] | None = None):
    from train_ppo import train_ppo
    from evaluate import build_ablation_configs, evaluate_across_regimes, to_ablation_summary_df

    ablation_cfgs = build_ablation_configs(CFG)
    regime_list = regimes or list(CFG.eval.regimes)
    all_results = {}

    for name, cfg in ablation_cfgs.items():
        print(f"\n{'=' * 60}\nAblation: {name}\n{'=' * 60}")
        hybrid_models = []
        for i in range(n_seeds):
            seed = cfg.train.seed + i * cfg.eval.train_seed_stride
            hybrid_models.append(
                train_ppo(
                    env_type="hybrid",
                    cfg=cfg,
                    seed=seed,
                    run_name=name,
                    use_lagrangian=cfg.ablation.use_cmdp,
                    verbose=1,
                    regime=regime_list[0],
                )
            )
        all_results[name] = evaluate_across_regimes(
            hybrid_model=hybrid_models,
            cfg=cfg,
            n_episodes=n_episodes,
            regimes=regime_list,
        )

    ablation_df = to_ablation_summary_df(all_results)
    with open(os.path.join(RESULTS, "ablation_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)
    ablation_df.to_csv(os.path.join(RESULTS, "table_ablation_summary.csv"), index=False)
    print("\n" + ablation_df.to_string(index=False))
    return all_results


def run_figures(results: dict | None = None):
    from figures import generate_figure_report

    if results is None:
        pkl = os.path.join(RESULTS, "results.pkl")
        if not os.path.exists(pkl):
            raise FileNotFoundError(f"No results file found: {pkl}")
        with open(pkl, "rb") as f:
            results = pickle.load(f)
    if results and isinstance(next(iter(results.values())), dict) and "mean_pnl" not in next(iter(results.values())):
        first_key = next(iter(results))
        results = results[first_key]

    pdf_path = generate_figure_report(results, out_dir=FIGURES)
    print(f"Saved figures to: {FIGURES}")
    print(f"PDF report: {pdf_path}")
    return pdf_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo", "train", "eval", "figures", "full", "ablation"], default="demo")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--env-steps", type=int, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--regimes", nargs="+", default=list(CFG.eval.regimes))
    return p.parse_args()


def main():
    args = parse_args()
    if args.env_steps is not None:
        CFG.env.n_steps = int(args.env_steps)
        print(f"[CONFIG] env.n_steps set to {CFG.env.n_steps}")

    if args.mode == "demo":
        print("Checking install ...")
        check_install()
        run_demo(episodes=args.episodes)
    elif args.mode == "train":
        _training_budget_summary(n_seeds=args.seeds, total_steps=args.steps, mode=args.mode, regimes=args.regimes)
        run_training(n_seeds=args.seeds, total_steps=args.steps)
    elif args.mode == "eval":
        run_eval(trained=None, n_episodes=args.episodes, regimes=args.regimes)
    elif args.mode == "figures":
        run_figures(results=None)
    elif args.mode == "ablation":
        _training_budget_summary(n_seeds=args.seeds, total_steps=args.steps, mode=args.mode, regimes=args.regimes)
        run_ablation_study(n_seeds=args.seeds, n_episodes=args.episodes, regimes=args.regimes)
    elif args.mode == "full":
        print("=" * 60)
        print("Full pipeline: train -> eval -> figures")
        _training_budget_summary(n_seeds=args.seeds, total_steps=args.steps, mode=args.mode, regimes=args.regimes)
        trained = run_training(n_seeds=args.seeds, total_steps=args.steps)
        regime_results = run_eval(trained=trained, n_episodes=args.episodes, regimes=args.regimes)
        run_figures(results=regime_results)
        print("=" * 60)


if __name__ == "__main__":
    main()
