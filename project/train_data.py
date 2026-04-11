"""
Train and evaluate Avellaneda-Stoikov, Pure RL, and Hybrid AS+RL
on real BTC minute-bar data with episode-wise price normalization.
"""

import os
import sys
from collections import deque
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

os.environ["OMP_NUM_THREADS"] = "1"

from agents.ppo import PPOConfig, PPOTrainer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from envs.lob_data_env import LOBDataEnv, VecLOBDataEnv
from models.avellaneda_stoikov import AvellanedaStoikov

OUTPUT_DIR = os.path.join(_HERE, "outputs")
DATA_PATH = "E:/CTA分钟级别/data/binance_btc_perp/btc_usdt_perp_1m_from_2020.csv"

TRAIN_PERIOD = ("2020-06-01", "2022-09-30")
TEST_PERIODS = {
    "COVID_Crash_2020Q1": ("2020-03-01", "2020-04-30"),
    "FTX_Collapse_2022Q4": ("2022-10-15", "2022-11-30"),
    "Bull_Run_2023Q1": ("2023-01-01", "2023-03-31"),
    "LowVol_Sideways_2023Q3": ("2023-07-01", "2023-09-30"),
    "ETF_Rally_2024Q1": ("2024-01-01", "2024-03-31"),
    "ATH_2024Q4": ("2024-10-01", "2025-01-31"),
}

ENV_KWARGS = dict(
    episode_steps=500,
    tick_size=0.01,
    max_inventory=10.0,
    initial_cash=1e6,
    fee_rate=2e-4,
    cancel_cost=1e-5,
    lambda_inventory=0.005,
    lambda_adverse=0.3,
    terminal_penalty=0.10,
    flatten_fee_rate=7e-4,
    shield_start_frac=0.60,
    shield_intensity=1.25,
    vol_window=20,
)

AS_PARAMS_BASE = dict(
    gamma=0.1,
    k=8.0,
    A=1.0,
    horizon=1.0,
    tick_size=0.01,
)

CALIBRATION_OFFSETS = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50], dtype=float)
AS_GAMMA_GRID = np.array([0.001, 0.002, 0.005, 0.010, 0.020], dtype=float)
AS_HORIZON_GRID = np.array([0.10, 0.25, 0.50, 1.00], dtype=float)
AS_K_MULTIPLIERS = np.array([0.50, 1.00, 1.50], dtype=float)

BASE_PPO_CFG = PPOConfig(
    obs_dim=12,
    action_dim=4,
    hidden_dim=256,
    n_hidden=2,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.999,
    gae_lambda=0.95,
    clip_eps=0.2,
    entropy_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    n_envs=8,
    rollout_steps=512,
    n_epochs=10,
    batch_size=256,
    total_timesteps=3_000_000,
    use_lagrangian=True,
    constraint_budget=0.10,  # 0.02 过严，初期库存随机性高，放松至 0.10
    eta_lr=0.01,              # 0.10 过快，eta 暴涨压垮 advantage；改为 0.01
    eta_max=2.0,              # 上限收紧，防止 eta 过大
    inventory_limit=ENV_KWARGS["max_inventory"],
    target_kl=0.05,  # 放宽早停阈值，由 logstd clamp 保证稳定
)


class StaticASAgent:
    def __init__(self, as_params):
        self.model = AvellanedaStoikov(**as_params)

    def act(self, _obs, mid, inventory, t_rem, sigma_hat):
        self.model.sigma = float(sigma_hat)
        bid, ask, _, _ = self.model.quotes(float(mid), float(inventory), float(t_rem))
        bid_off = max(float(mid) - bid, self.model.tick_size)
        ask_off = max(ask - float(mid), self.model.tick_size)
        return np.array([np.log(bid_off), np.log(ask_off), 0.0, 0.0], dtype=np.float32)


def load_period(df_full: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df_full["dt"] >= start) & (df_full["dt"] <= end)
    sub = df_full[mask].reset_index(drop=True)
    assert len(sub) > ENV_KWARGS["episode_steps"] * 10, f"Period {start}~{end} too short: {len(sub)} rows"
    return sub


def compute_metrics(rewards, inv, bid_f, ask_f, adv, terminal_pnl, max_inv=50.0):
    rewards = np.asarray(rewards, dtype=float)
    cum = np.cumsum(rewards)
    n_tail = max(int(np.ceil(0.05 * len(rewards))), 1) if len(rewards) else 1
    downside = rewards[rewards < 0]

    return {
        "net_pnl": float(terminal_pnl),
        "reward_sum": float(cum[-1]) if len(cum) else 0.0,
        "sharpe": float(rewards.mean() / rewards.std() * np.sqrt(len(rewards))) if len(rewards) and rewards.std() > 1e-10 else 0.0,
        "sortino": float(rewards.mean() / downside.std() * np.sqrt(len(rewards))) if len(downside) and downside.std() > 1e-10 else 0.0,
        "max_drawdown": float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0,
        "cvar95": float(-np.sort(rewards)[:n_tail].mean()) if len(rewards) else 0.0,
        "mean_inventory": float(np.mean(np.abs(inv))) if len(inv) else 0.0,
        "fill_bid": float(np.mean(bid_f)) if len(bid_f) else 0.0,
        "fill_ask": float(np.mean(ask_f)) if len(ask_f) else 0.0,
        "adv_sel": float(np.mean(adv)) if len(adv) else 0.0,
        "time_at_limit": float(np.mean(np.abs(inv) >= max_inv * 0.9)) if len(inv) else 0.0,
    }


def _offset_action(offset: float, size: float = 1.0) -> np.ndarray:
    off = max(float(offset), ENV_KWARGS["tick_size"])
    log_size = np.log(float(size))
    return np.array([np.log(off), np.log(off), log_size, log_size], dtype=np.float32)


def _aggregate_episode_metrics(metrics: list[dict]) -> dict:
    keys = metrics[0].keys()
    return {key: float(np.mean([m[key] for m in metrics])) for key in keys}


def _as_score(metrics: dict) -> float:
    fill_avg = 0.5 * (metrics["fill_bid"] + metrics["fill_ask"])
    pnl_term = np.sign(metrics["net_pnl"]) * np.log1p(abs(metrics["net_pnl"]))
    # 目标成交率：bar range 中位数0.095，half-spread应在0.01~0.10之间（fill 10%~96%）
    # 成交率<5% 意味着几乎不参与市场，硬性惩罚确保不选到宽价差策略
    MIN_FILL = 0.05
    fill_penalty = max(MIN_FILL - fill_avg, 0.0) / MIN_FILL  # 0→1 as fill→0
    return (
        metrics["sharpe"]
        + 0.10 * pnl_term
        + 0.50 * fill_avg
        - 0.50 * metrics["time_at_limit"]
        - 3.0 * fill_penalty   # 强惩罚：零成交时总分 -3，强制选有成交的配置
    )


def fit_fill_curve(df_train: pd.DataFrame, vol_thresholds, n_episodes: int = 24) -> tuple[pd.DataFrame, float, float]:
    env = LOBDataEnv(df=df_train, seed=0, vol_thresholds=vol_thresholds, **ENV_KWARGS)
    max_start = len(df_train) - ENV_KWARGS["episode_steps"] - 2
    starts = np.linspace(0, max_start, n_episodes, dtype=int)
    rows = []

    for offset in CALIBRATION_OFFSETS:
        action = _offset_action(offset)
        bid_fills, ask_fills = [], []
        for start in starts:
            env.reset(start_idx=int(start))
            done = False
            while not done:
                _, _, term, trunc, info = env.step(action)
                done = term or trunc
                bid_fills.append(float(info.bid_filled))
                ask_fills.append(float(info.ask_filled))

        p_bid = float(np.mean(bid_fills)) if bid_fills else 0.0
        p_ask = float(np.mean(ask_fills)) if ask_fills else 0.0
        p_side = 0.5 * (p_bid + p_ask)
        intensity = float(-np.log(np.clip(1.0 - p_side, 1e-6, 1.0)))
        rows.append(
            {
                "offset": float(offset),
                "fill_bid": p_bid,
                "fill_ask": p_ask,
                "fill_side": p_side,
                "intensity": intensity,
            }
        )

    df_curve = pd.DataFrame(rows)
    fit_df = df_curve[df_curve["intensity"] > 1e-6].copy()
    coeff = np.polyfit(fit_df["offset"].values, np.log(fit_df["intensity"].values), deg=1)
    k_fit = max(float(-coeff[0]), 1e-6)
    A_fit = float(np.exp(coeff[1]))
    df_curve["log_intensity_fit"] = coeff[1] - k_fit * df_curve["offset"]
    df_curve["intensity_fit"] = np.exp(df_curve["log_intensity_fit"])
    df_curve["fill_fit"] = 1.0 - np.exp(-df_curve["intensity_fit"])
    return df_curve, k_fit, A_fit


def calibrate_as_params(df_train: pd.DataFrame, sigma_eff: float, vol_thresholds):
    fill_curve, k_fit, a_fit = fit_fill_curve(df_train, vol_thresholds)
    env = LOBDataEnv(df=df_train, seed=0, vol_thresholds=vol_thresholds, **ENV_KWARGS)
    env.sigma_base = sigma_eff
    max_start = len(df_train) - ENV_KWARGS["episode_steps"] - 2
    starts = np.linspace(0, max_start, 12, dtype=int)

    rows = []
    for gamma in AS_GAMMA_GRID:
        for k_mult in AS_K_MULTIPLIERS:
            k_val = max(k_fit * float(k_mult), 1e-6)
            for horizon in AS_HORIZON_GRID:
                as_params = {
                    **AS_PARAMS_BASE,
                    "gamma": float(gamma),
                    "k": k_val,
                    "A": a_fit,
                    "horizon": float(horizon),
                    "sigma": sigma_eff,
                }
                agent = StaticASAgent(as_params)
                metrics = [run_episode(agent, env, int(start), sigma_eff) for start in starts]
                agg = _aggregate_episode_metrics(metrics)
                tau0 = as_params["horizon"]
                opening_spread = (
                    as_params["gamma"] * sigma_eff ** 2 * tau0
                    + (2.0 / as_params["gamma"]) * np.log(1.0 + as_params["gamma"] / as_params["k"])
                )
                rows.append(
                    {
                        "gamma": as_params["gamma"],
                        "k": as_params["k"],
                        "A": as_params["A"],
                        "horizon": as_params["horizon"],
                        "opening_spread": float(opening_spread),
                        "score": _as_score(agg),
                        **agg,
                    }
                )

    grid = pd.DataFrame(rows).sort_values(["score", "sharpe", "net_pnl"], ascending=False).reset_index(drop=True)
    best = grid.iloc[0]
    as_params = {
        **AS_PARAMS_BASE,
        "gamma": float(best["gamma"]),
        "k": float(best["k"]),
        "A": a_fit,
        "horizon": float(best["horizon"]),
        "sigma": sigma_eff,
    }
    summary = pd.DataFrame(
        [
            {"item": "sigma_eff", "value": float(sigma_eff)},
            {"item": "k_fit_from_fill_curve", "value": float(k_fit)},
            {"item": "A_fit_from_fill_curve", "value": float(a_fit)},
            {"item": "best_gamma", "value": float(best["gamma"])},
            {"item": "best_k", "value": float(best["k"])},
            {"item": "best_horizon", "value": float(best["horizon"])},
            {"item": "best_score", "value": float(best["score"])},
            {"item": "best_sharpe", "value": float(best["sharpe"])},
            {"item": "best_net_pnl", "value": float(best["net_pnl"])},
            {"item": "best_fill_bid", "value": float(best["fill_bid"])},
            {"item": "best_fill_ask", "value": float(best["fill_ask"])},
        ]
    )
    return as_params, fill_curve, grid, summary


def save_calibration_tables(fill_curve: pd.DataFrame, grid: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> None:
    fill_csv = os.path.join(out_dir, "as_fill_curve.csv")
    grid_csv = os.path.join(out_dir, "as_grid_search.csv")
    summary_csv = os.path.join(out_dir, "as_calibration_summary.csv")
    summary_tex = os.path.join(out_dir, "as_calibration_summary.tex")
    top_grid_tex = os.path.join(out_dir, "as_grid_top10.tex")

    fill_curve.to_csv(fill_csv, index=False)
    grid.to_csv(grid_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary.to_latex(summary_tex, index=False, escape=False)
    grid.head(10).to_latex(top_grid_tex, index=False, escape=False, float_format=lambda x: f"{x:.4f}")

    print(f"  Saved -> {fill_csv}")
    print(f"  Saved -> {grid_csv}")
    print(f"  Saved -> {summary_csv}")
    print(f"  Saved -> {summary_tex}")
    print(f"  Saved -> {top_grid_tex}")


def build_trainers(as_params: dict):

    hybrid_cfg = replace(BASE_PPO_CFG, policy_mode="hybrid")
    pure_cfg = replace(BASE_PPO_CFG, policy_mode="model_free")

    trainers = {
        "Pure RL": {
            "trainer": PPOTrainer(pure_cfg, backbone=None),
            "cfg": pure_cfg,
            "ckpt_prefix": "pure_rl",
            "curve_path": os.path.join(OUTPUT_DIR, "data_pure_rl_training_curves.png"),
            "model_path": os.path.join(OUTPUT_DIR, "data_pure_rl_final.pt"),
        },
        "Hybrid AS+RL": {
            "trainer": PPOTrainer(hybrid_cfg, backbone=AvellanedaStoikov(**as_params)),
            "cfg": hybrid_cfg,
            "ckpt_prefix": "hybrid",
            "curve_path": os.path.join(OUTPUT_DIR, "data_hybrid_training_curves.png"),
            "model_path": os.path.join(OUTPUT_DIR, "data_hybrid_final.pt"),
        },
    }
    return trainers, as_params


def train_one(label: str, trainer: PPOTrainer, cfg: PPOConfig, df_train: pd.DataFrame, vol_thresholds, sigma_eff: float, ckpt_prefix: str):
    print("=" * 72)
    print(f"  {label} | Real BTC Minute Data")
    print(f"  policy_mode={cfg.policy_mode}  device={cfg.device}  steps={cfg.total_timesteps:,}")
    print("=" * 72)

    vec_env = VecLOBDataEnv(cfg.n_envs, df_train, ENV_KWARGS, vol_thresholds=vol_thresholds)
    obs = vec_env.reset(seed=42)
    dones = np.zeros(cfg.n_envs, dtype=bool)
    ep_rew_buf = np.zeros(cfg.n_envs)
    ep_rew_q = deque(maxlen=100)

    history = {k: [] for k in ["steps", "ep_rew", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "eta", "constraint"]}
    total_steps = 0
    n_updates = cfg.total_timesteps // (cfg.rollout_steps * cfg.n_envs)
    pbar = tqdm(total=cfg.total_timesteps, desc=label, unit="step")

    for update in range(1, n_updates + 1):
        for _ in range(cfg.rollout_steps):
            inv_arr = obs[:, 1] * ENV_KWARGS["max_inventory"]
            t_arr = obs[:, 6]
            mid_arr = LOBDataEnv.S0_NORM * (1.0 + obs[:, 0])
            # sigma 传价格单位（S0=100空间）：obs[:,5] = sigma_hat/sigma_base
            # sig_arr = rolling_vol * S0_NORM，使 AS 库存调整项 γσ²τ 有意义
            sig_arr = np.abs(obs[:, 5]) * sigma_eff * LOBDataEnv.S0_NORM + 1e-8

            decoded_actions, raw_actions, logps, vals = trainer.act(obs, mid_arr, inv_arr, t_arr, sig_arr)
            next_obs, rewards, dones_step, _ = vec_env.step(decoded_actions)

            inv_next = np.abs(next_obs[:, 1])
            costs = inv_next ** 2
            # buffer 存 raw_actions（网络输出），保证 update 时 log_prob 计算一致
            trainer.buffer.add(obs, raw_actions, rewards, costs, vals, logps, dones_step)

            obs = next_obs
            dones = dones_step
            total_steps += cfg.n_envs
            pbar.update(cfg.n_envs)

            ep_rew_buf += rewards
            for i, done in enumerate(dones_step):
                if done:
                    ep_rew_q.append(ep_rew_buf[i])
                    ep_rew_buf[i] = 0.0

        stats = trainer.update(obs, dones)
        trainer.global_step = total_steps
        ep_mean = float(np.mean(ep_rew_q)) if ep_rew_q else 0.0
        history["steps"].append(total_steps)
        history["ep_rew"].append(ep_mean)
        for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "eta", "constraint"]:
            history[key].append(stats.get(key, 0.0))

        pbar.set_postfix({"rew": f"{ep_mean:.2f}", "KL": f"{stats.get('approx_kl', 0.0):.4f}", "eta": f"{stats.get('eta', 0.0):.4f}"})

        if update % 50 == 0:
            trainer.save(os.path.join(OUTPUT_DIR, f"data_{ckpt_prefix}_ckpt_{update:04d}.pt"))

    pbar.close()
    return history


def run_episode(agent, env: LOBDataEnv, start_idx: int, sigma_eff: float):
    obs, _ = env.reset(start_idx=start_idx)
    rewards, inv, bid_f, ask_f, adv = [], [], [], [], []
    done = False

    while not done:
        mid = env.mid
        inv_now = env.inventory
        t_rem = (env.episode_steps - env.step_idx) / env.episode_steps
        # sigma 价格单位：与 train_one 中的 sig_arr 保持一致
        sig = float(env.sigma_hat / (env.sigma_base + 1e-8)) * sigma_eff * LOBDataEnv.S0_NORM

        if isinstance(agent, PPOTrainer):
            decoded, _, _, _ = agent.act(
                obs[np.newaxis, :],
                np.array([mid]),
                np.array([inv_now]),
                np.array([t_rem]),
                np.array([sig]),
                deterministic=True,
            )
            act = decoded.squeeze(0).astype(np.float32)
        else:
            act = np.asarray(agent.act(obs, mid, inv_now, t_rem, sig), dtype=np.float32)

        obs, reward, term, trunc, info = env.step(act)
        done = term or trunc
        rewards.append(reward)
        inv.append(inv_now)
        bid_f.append(info.bid_filled)
        ask_f.append(info.ask_filled)
        adv.append(info.adverse_selection)

    return compute_metrics(
        np.asarray(rewards, dtype=float),
        np.asarray(inv, dtype=float),
        np.asarray(bid_f, dtype=float),
        np.asarray(ask_f, dtype=float),
        np.asarray(adv, dtype=float),
        terminal_pnl=float(env.portfolio_value - env.initial_cash),
        max_inv=env.max_inventory,
    )


def evaluate_period(period_name: str, df_period: pd.DataFrame, agents: dict, sigma_eff: float, vol_thresholds, n_episodes: int = 20):
    env = LOBDataEnv(df=df_period, seed=0, vol_thresholds=vol_thresholds, **ENV_KWARGS)
    env.sigma_base = sigma_eff

    max_start = len(df_period) - ENV_KWARGS["episode_steps"] - 2
    starts = np.linspace(0, max_start, n_episodes, dtype=int)
    rows = []

    for label, agent in agents.items():
        metrics = []
        for start in starts:
            try:
                metrics.append(run_episode(agent, env, int(start), sigma_eff))
            except Exception:
                continue
        if not metrics:
            continue

        rows.append(
            {
                "period": period_name,
                "agent": label,
                "n_episodes": len(metrics),
                "pnl_mean": float(np.mean([m["net_pnl"] for m in metrics])),
                "pnl_std": float(np.std([m["net_pnl"] for m in metrics])),
                "sharpe": float(np.mean([m["sharpe"] for m in metrics])),
                "sortino": float(np.mean([m["sortino"] for m in metrics])),
                "max_dd": float(np.mean([m["max_drawdown"] for m in metrics])),
                "cvar95": float(np.mean([m["cvar95"] for m in metrics])),
                "inv_mean": float(np.mean([m["mean_inventory"] for m in metrics])),
                "fill_bid": float(np.mean([m["fill_bid"] for m in metrics])),
                "fill_ask": float(np.mean([m["fill_ask"] for m in metrics])),
                "adv_sel": float(np.mean([m["adv_sel"] for m in metrics])),
                "time_at_limit": float(np.mean([m["time_at_limit"] for m in metrics])),
            }
        )
    return rows


def smooth(x, w=10):
    arr = np.asarray(x, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode="valid") if len(arr) >= w else arr


def plot_training(history, path, title):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    panels = [
        ("ep_rew", "Episode Reward", "royalblue"),
        ("policy_loss", "Policy Loss", "darkorange"),
        ("value_loss", "Value Loss", "green"),
        ("entropy", "Entropy", "purple"),
        ("approx_kl", "Approx KL", "brown"),
        ("clip_frac", "Clip Fraction", "teal"),
        ("eta", "Lagrangian eta", "crimson"),
        ("constraint", "Constraint Cost", "darkgreen"),
    ]

    steps = history["steps"]
    for ax, (key, panel_title, color) in zip(axes.ravel(), panels):
        series = smooth(history[key])
        xs = steps[: len(series)]
        ax.plot(xs, series, color=color, lw=1.6)
        ax.fill_between(xs, series, alpha=0.1, color=color)
        ax.set_title(panel_title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Steps", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    axes.ravel()[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def plot_period_results(df_res: pd.DataFrame, path):
    periods = list(df_res["period"].unique())
    agents = ["Static AS", "Pure RL", "Hybrid AS+RL"]
    colors = ["#FF9800", "#4CAF50", "#2196F3"]
    metrics = [("pnl_mean", "Net P&L"), ("sharpe", "Sharpe"), ("max_dd", "Max DD"), ("inv_mean", "Mean |Inv|")]

    fig, axes = plt.subplots(len(metrics), len(periods), figsize=(3.6 * len(periods), 3.5 * len(metrics)))
    fig.suptitle("Real BTC Minute Data: AS vs Pure RL vs Hybrid AS+RL", fontsize=12, fontweight="bold")

    for ci, period in enumerate(periods):
        sub = df_res[df_res["period"] == period]
        for ri, (metric, label) in enumerate(metrics):
            ax = axes[ri, ci] if len(periods) > 1 else axes[ri]
            vals = [sub.loc[sub["agent"] == agent, metric].iloc[0] if any(sub["agent"] == agent) else 0.0 for agent in agents]
            errs = [sub.loc[sub["agent"] == agent, "pnl_std"].iloc[0] if metric == "pnl_mean" and any(sub["agent"] == agent) else 0.0 for agent in agents]
            bars = ax.bar(agents, vals, yerr=errs if metric == "pnl_mean" else None, color=colors, capsize=4, alpha=0.85)
            if ci == 0:
                ax.set_ylabel(label, fontsize=9)
            if ri == 0:
                ax.set_title(period.replace("_", "\n"), fontsize=8, fontweight="bold")
            ax.tick_params(axis="x", labelsize=7, rotation=25)
            ax.grid(True, axis="y", alpha=0.3)
            for bar, val in zip(bars, vals):
                offset = 1.03 if val >= 0 else 0.97
                ax.text(bar.get_x() + bar.get_width() / 2, val * offset, f"{val:.2f}", ha="center", fontsize=6, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


def _paper_overall_table(df_res: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df_res.groupby("agent", as_index=False)
        .agg(
            periods=("period", "count"),
            pnl_mean=("pnl_mean", "mean"),
            pnl_std=("pnl_mean", "std"),
            sharpe=("sharpe", "mean"),
            sortino=("sortino", "mean"),
            max_dd=("max_dd", "mean"),
            cvar95=("cvar95", "mean"),
            inv_mean=("inv_mean", "mean"),
            fill_bid=("fill_bid", "mean"),
            fill_ask=("fill_ask", "mean"),
            adv_sel=("adv_sel", "mean"),
            time_at_limit=("time_at_limit", "mean"),
        )
    )
    grouped["PnL"] = grouped.apply(
        lambda row: f"{row['pnl_mean']:.2f} ± {0.0 if pd.isna(row['pnl_std']) else row['pnl_std']:.2f}",
        axis=1,
    )
    grouped["Sharpe"] = grouped["sharpe"].map(lambda x: f"{x:.3f}")
    grouped["Sortino"] = grouped["sortino"].map(lambda x: f"{x:.3f}")
    grouped["MaxDD"] = grouped["max_dd"].map(lambda x: f"{x:.2f}")
    grouped["CVaR95"] = grouped["cvar95"].map(lambda x: f"{x:.2f}")
    grouped["Mean|Inv|"] = grouped["inv_mean"].map(lambda x: f"{x:.2f}")
    grouped["BidFill"] = grouped["fill_bid"].map(lambda x: f"{x:.3f}")
    grouped["AskFill"] = grouped["fill_ask"].map(lambda x: f"{x:.3f}")
    grouped["AdvSel"] = grouped["adv_sel"].map(lambda x: f"{x:.4f}")
    grouped["Time@Limit"] = grouped["time_at_limit"].map(lambda x: f"{100.0 * x:.2f}\\%")
    return grouped[
        [
            "agent",
            "periods",
            "PnL",
            "Sharpe",
            "Sortino",
            "MaxDD",
            "CVaR95",
            "Mean|Inv|",
            "BidFill",
            "AskFill",
            "AdvSel",
            "Time@Limit",
        ]
    ].rename(columns={"agent": "Method", "periods": "Periods"})


def _paper_period_table(df_res: pd.DataFrame) -> pd.DataFrame:
    table = df_res.copy()
    table["PnL"] = table.apply(lambda row: f"{row['pnl_mean']:.2f} ± {row['pnl_std']:.2f}", axis=1)
    table["Sharpe"] = table["sharpe"].map(lambda x: f"{x:.3f}")
    table["Sortino"] = table["sortino"].map(lambda x: f"{x:.3f}")
    table["MaxDD"] = table["max_dd"].map(lambda x: f"{x:.2f}")
    table["CVaR95"] = table["cvar95"].map(lambda x: f"{x:.2f}")
    table["Mean|Inv|"] = table["inv_mean"].map(lambda x: f"{x:.2f}")
    table["BidFill"] = table["fill_bid"].map(lambda x: f"{x:.3f}")
    table["AskFill"] = table["fill_ask"].map(lambda x: f"{x:.3f}")
    table["AdvSel"] = table["adv_sel"].map(lambda x: f"{x:.4f}")
    table["Time@Limit"] = table["time_at_limit"].map(lambda x: f"{100.0 * x:.2f}\\%")
    return table[
        [
            "period",
            "agent",
            "PnL",
            "Sharpe",
            "Sortino",
            "MaxDD",
            "CVaR95",
            "Mean|Inv|",
            "BidFill",
            "AskFill",
            "AdvSel",
            "Time@Limit",
        ]
    ].rename(columns={"period": "Period", "agent": "Method"})


def save_paper_tables(df_res: pd.DataFrame, out_dir: str) -> None:
    overall = _paper_overall_table(df_res)
    by_period = _paper_period_table(df_res)

    overall_csv = os.path.join(out_dir, "paper_table_overall.csv")
    overall_tex = os.path.join(out_dir, "paper_table_overall.tex")
    period_csv = os.path.join(out_dir, "paper_table_by_period.csv")
    period_tex = os.path.join(out_dir, "paper_table_by_period.tex")

    overall.to_csv(overall_csv, index=False)
    by_period.to_csv(period_csv, index=False)
    overall.to_latex(overall_tex, index=False, escape=False)
    by_period.to_latex(period_tex, index=False, escape=False)

    print(f"  Saved -> {overall_csv}")
    print(f"  Saved -> {overall_tex}")
    print(f"  Saved -> {period_csv}")
    print(f"  Saved -> {period_tex}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading BTC minute data...")
    df_full = pd.read_csv(DATA_PATH)
    df_full["dt"] = pd.to_datetime(df_full["timestamp"], unit="ms")
    print(f"  Total rows: {len(df_full):,}  ({df_full['dt'].iloc[0].date()} ~ {df_full['dt'].iloc[-1].date()})")

    df_train = load_period(df_full, *TRAIN_PERIOD)
    print(f"  Train period: {TRAIN_PERIOD[0]} ~ {TRAIN_PERIOD[1]}  rows={len(df_train):,}")

    mid_train = (df_train["high"] + df_train["low"]).values / 2.0
    ret_train = pd.Series(mid_train).pct_change().dropna().values
    sigma_step = float(ret_train.std())
    sigma_eff = sigma_step * np.sqrt(ENV_KWARGS["episode_steps"])
    print(f"  sigma_step={sigma_step:.6f}  sigma_eff={sigma_eff:.6f}")

    tmp_env = LOBDataEnv(df=df_train, seed=0, **ENV_KWARGS)
    vol_thresholds = (tmp_env._vol_lo, tmp_env._vol_hi)
    print(f"  Regime thresholds: lo={vol_thresholds[0]:.6f}  hi={vol_thresholds[1]:.6f}")
    print(f"  Episode normalization: real price path is rescaled to S0={LOBDataEnv.S0_NORM:.1f} inside each episode")

    print("\nCalibrating AS benchmark on training data...")
    as_params, fill_curve, as_grid, as_summary = calibrate_as_params(df_train, sigma_eff, vol_thresholds)
    save_calibration_tables(fill_curve, as_grid, as_summary, OUTPUT_DIR)
    print(
        "  Calibrated AS: "
        f"gamma={as_params['gamma']:.4f}  k={as_params['k']:.4f}  "
        f"A={as_params['A']:.4f}  horizon={as_params['horizon']:.4f}"
    )

    trainers, as_params = build_trainers(as_params)
    for label, spec in trainers.items():
        history = train_one(label, spec["trainer"], spec["cfg"], df_train, vol_thresholds, sigma_eff, spec["ckpt_prefix"])
        spec["trainer"].save(spec["model_path"])
        plot_training(history, spec["curve_path"], f"{label} Training on Real BTC Minute Data")

    agents = {
        "Static AS": StaticASAgent(as_params),
        "Pure RL": trainers["Pure RL"]["trainer"],
        "Hybrid AS+RL": trainers["Hybrid AS+RL"]["trainer"],
    }

    print("\n" + "=" * 72)
    print("  Out-of-Sample Evaluation on Real BTC Minute Data")
    print("=" * 72)

    all_rows = []
    for period_name, (start, end) in TEST_PERIODS.items():
        try:
            df_period = load_period(df_full, start, end)
        except AssertionError as err:
            print(f"  Skipping {period_name}: {err}")
            continue

        rows = evaluate_period(period_name, df_period, agents, sigma_eff, vol_thresholds, n_episodes=20)
        all_rows.extend(rows)

        print(f"\n[{period_name}]  {start} ~ {end}")
        print(f"{'Agent':<16} {'PnL':>10} {'Sharpe':>8} {'MaxDD':>8} {'|Inv|':>8} {'BidFill':>8}")
        for row in rows:
            print(
                f"{row['agent']:<16} {row['pnl_mean']:>+10.2f} {row['sharpe']:>8.3f} "
                f"{row['max_dd']:>8.2f} {row['inv_mean']:>8.2f} {row['fill_bid']:>8.3f}"
            )

    df_res = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, "data_period_results.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\n  Results saved -> {csv_path}")

    save_paper_tables(df_res, OUTPUT_DIR)
    plot_period_results(df_res, os.path.join(OUTPUT_DIR, "data_period_comparison.png"))
    print(f"\n  All outputs saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
