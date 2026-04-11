"""
Synthetic LOB Simulation — Training Script
===========================================
Trains Pure RL (Model-Free PPO) and Hybrid AS+RL on the LOBEnv simulator.
Evaluates against Static AS and Constrained AS baselines.

Fixes vs. original:
  1. act() now returns (decoded, raw, logps, vals) — unpack correctly
  2. Buffer stores raw (tanh-space) actions, NOT decoded log-offsets
  3. sig_arr uses episode-level sigma (SIGMA_EFF ≈ 0.447), matching static AS
  4. Trains both Pure RL + Hybrid AS+RL for paper comparison
"""

import os
import sys
from collections import deque
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from agents.ppo import PPOTrainer, PPOConfig
from envs.lob_env import LOBEnv, VecLOBEnv
from models.avellaneda_stoikov import AvellanedaStoikov
from utils.metrics import compute_metrics, StaticASAgent

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(_HERE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENV_KWARGS = dict(
    S0=100.0, sigma=0.02, mu=0.0,
    jump_prob=0.005, jump_scale=0.5,
    tick_size=0.01, episode_steps=500,
    max_inventory=30, initial_cash=1e6,
    base_intensity=5.0, k_flow=1.5,
    fee_rate=2e-4, cancel_cost=1e-5,
    lambda_inventory=0.005, lambda_adverse=0.3,
    use_regimes=True,
)

# Episode-level sigma in log-return units: σ_eff = σ_step × √N
SIGMA_EFF = ENV_KWARGS["sigma"] * np.sqrt(ENV_KWARGS["episode_steps"])   # ≈ 0.447

AS_PARAMS = dict(
    gamma=0.1, sigma=SIGMA_EFF, k=1.5,
    A=5.0, horizon=1.0, tick_size=0.01,
)

BASE_CFG = PPOConfig(
    obs_dim=12, action_dim=4,
    hidden_dim=256, n_hidden=2,
    lr_actor=3e-4, lr_critic=1e-3,
    gamma=0.999, gae_lambda=0.95,
    clip_eps=0.2, entropy_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5,
    n_envs=8, rollout_steps=512,
    n_epochs=10, batch_size=256,
    total_timesteps=3_000_000,
    use_lagrangian=True,
    constraint_budget=0.10,
    eta_lr=0.01,
    eta_max=2.0,
    target_kl=0.05,
    inventory_limit=30,
)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_agent(cfg, backbone, label, ckpt_prefix):
    print(f"\n{'='*60}")
    print(f"  Training: {label}")
    print(f"  policy_mode={cfg.policy_mode}  device={cfg.device}")
    print(f"  total_steps={cfg.total_timesteps:,}")
    print(f"{'='*60}")

    trainer = PPOTrainer(cfg, backbone=backbone)
    vec_env = VecLOBEnv(cfg.n_envs, ENV_KWARGS)
    obs = vec_env.reset(seed=42)
    dones = np.zeros(cfg.n_envs, dtype=bool)
    ep_rew_buf = np.zeros(cfg.n_envs)
    ep_rew_q = deque(maxlen=100)

    history = {k: [] for k in ["steps", "ep_rew", "policy_loss", "value_loss",
                                 "entropy", "approx_kl", "clip_frac", "eta", "constraint"]}
    total_steps = 0
    n_updates = cfg.total_timesteps // (cfg.rollout_steps * cfg.n_envs)
    pbar = tqdm(total=cfg.total_timesteps, desc=label, unit="step")

    for update in range(1, n_updates + 1):
        for _ in range(cfg.rollout_steps):
            # Extract AS backbone inputs from obs
            inv_arr = obs[:, 1] * ENV_KWARGS["max_inventory"]        # absolute inventory
            t_arr   = obs[:, 6]                                        # normalized time remaining
            mid_arr = (obs[:, 0] + 1.0) * ENV_KWARGS["S0"]           # absolute mid-price
            # sigma in episode-level log-return units (matches AS_PARAMS.sigma)
            sig_arr = np.abs(obs[:, 5]) * SIGMA_EFF + 1e-8

            # act() returns (decoded_actions, raw_actions, logps, vals)
            decoded, raw, logps, vals = trainer.act(
                obs, mid_arr, inv_arr, t_arr, sig_arr)

            # Send decoded (log-offset space) to env
            next_obs, rewards, dones_step, infos = vec_env.step(decoded)

            # CMDP cost = normalised inventory cost = (I/I_max)^2 = obs[:,1]^2
            costs = obs[:, 1] ** 2

            # CRITICAL: store raw tanh-space actions for correct PPO log_prob
            trainer.buffer.add(obs, raw, rewards, costs, vals, logps, dones_step)

            obs = next_obs
            dones = dones_step
            total_steps += cfg.n_envs
            pbar.update(cfg.n_envs)

            ep_rew_buf += rewards
            for i, d in enumerate(dones_step):
                if d:
                    ep_rew_q.append(ep_rew_buf[i])
                    ep_rew_buf[i] = 0.0

        stats = trainer.update(obs, dones)
        trainer.global_step = total_steps

        ep_mean = float(np.mean(ep_rew_q)) if ep_rew_q else 0.0
        history["steps"].append(total_steps)
        history["ep_rew"].append(ep_mean)
        for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "eta", "constraint"]:
            history[k].append(stats.get(k, 0.0))

        pbar.set_postfix({
            "rew": f"{ep_mean:.2f}",
            "KL":  f"{stats.get('approx_kl', 0.0):.4f}",
            "eta": f"{stats.get('eta', 0.0):.4f}",
        })

        if update % 50 == 0:
            trainer.save(os.path.join(OUTPUT_DIR, f"{ckpt_prefix}_ckpt_{update:04d}.pt"))

    pbar.close()
    trainer.save(os.path.join(OUTPUT_DIR, f"{ckpt_prefix}_final.pt"))
    print(f"  {label} done — {total_steps:,} steps")
    return trainer, history


# ─────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────

class ConstrainedASAgent:
    """AS with hard inventory clip at ±0.8 × max_inv before quoting."""
    def __init__(self, as_params, max_inv=30, clip_frac=0.8):
        self.m = AvellanedaStoikov(**as_params)
        self.clip = clip_frac * max_inv

    def act(self, obs, mid, inventory, t_rem, sigma_hat=None, **kw):
        inv_clipped = float(np.clip(inventory, -self.clip, self.clip))
        sigma = sigma_hat if sigma_hat is not None else self.m.sigma
        bid, ask, _, _ = self.m.quotes(mid, inv_clipped, t_rem, sigma)
        bo = max(mid - bid, self.m.tick_size)
        ao = max(ask - mid, self.m.tick_size)
        return np.array([np.log(bo), np.log(ao), 0.0, 0.0], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def run_episode(agent, env, sigma_eff):
    obs, _ = env.reset()
    pnl, inv, bf, af, adv = [], [], [], [], []
    done = False
    while not done:
        mid  = env.mid
        invv = env.inventory
        t    = (env.episode_steps - env.step_idx) / env.episode_steps
        sig  = float(abs(obs[5])) * sigma_eff + 1e-8

        if isinstance(agent, PPOTrainer):
            decoded, _, _, _ = agent.act(
                obs[np.newaxis], np.array([mid]),
                np.array([invv]), np.array([t]),
                np.array([sig]), deterministic=True)
            a = decoded.squeeze(0).astype(np.float32)
        else:
            a = np.asarray(agent.act(obs, mid, invv, t, sig), dtype=np.float32)
            if a.ndim == 2:
                a = a[0]

        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        pnl.append(r)
        inv.append(invv)
        bf.append(info.bid_filled)
        af.append(info.ask_filled)
        adv.append(info.adverse_selection)

    return compute_metrics(
        np.array(pnl), np.array(inv, dtype=float),
        np.array(bf), np.array(af), np.array(adv),
        max_inv=env.max_inventory,
    )


def evaluate_all(agents, n_episodes=20, sigma_eff=SIGMA_EFF):
    print(f"\n{'='*60}")
    print(f"  Evaluation ({n_episodes} episodes per agent)")
    print(f"{'='*60}")

    all_metrics = {}
    for name, agent in agents.items():
        ms = []
        for seed in range(n_episodes):
            env = LOBEnv(**{**ENV_KWARGS, "seed": seed * 13 + 7})
            ms.append(run_episode(agent, env, sigma_eff))

        r = {
            "pnl":      np.array([m.net_pnl        for m in ms]),
            "sharpe":   np.array([m.sharpe          for m in ms]),
            "sortino":  np.array([m.sortino         for m in ms]),
            "mdd":      np.array([m.max_drawdown    for m in ms]),
            "cvar":     np.array([m.cvar_95         for m in ms]),
            "inv":      np.array([m.mean_inventory  for m in ms]),
            "fill_bid": np.array([m.fill_rate_bid   for m in ms]),
            "fill_ask": np.array([m.fill_rate_ask   for m in ms]),
        }
        all_metrics[name] = r
        print(f"  {name:<22s}  PnL={r['pnl'].mean():+8.3f}±{r['pnl'].std():.3f}"
              f"  Sharpe={r['sharpe'].mean():6.3f}"
              f"  MDD={r['mdd'].mean():.3f}"
              f"  Fill={r['fill_bid'].mean():.3f}")

    return all_metrics


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────

def _smooth(x, w=8):
    a = np.array(x, dtype=float)
    return np.convolve(a, np.ones(w) / w, mode="valid") if len(a) >= w else a


def plot_training_curves(histories, path):
    """Plot training curves for all RL agents side by side."""
    n_agents = len(histories)
    fig, axes = plt.subplots(n_agents, 8, figsize=(24, 4 * n_agents))
    if n_agents == 1:
        axes = axes[np.newaxis, :]

    panels = [
        ("ep_rew",      "Episode Reward",   "royalblue"),
        ("policy_loss", "Policy Loss",      "darkorange"),
        ("value_loss",  "Value Loss",       "green"),
        ("entropy",     "Entropy",          "purple"),
        ("approx_kl",   "Approx KL",       "brown"),
        ("clip_frac",   "Clip Fraction",    "teal"),
        ("eta",         "Lagrangian η",     "crimson"),
        ("constraint",  "Constraint Cost",  "darkgreen"),
    ]

    for row, (label, history) in enumerate(histories.items()):
        steps = history["steps"]
        for col, (key, title, color) in enumerate(panels):
            ax = axes[row, col]
            sm = _smooth(history[key])
            xs = steps[:len(sm)]
            ax.plot(xs, sm, color=color, lw=1.5)
            ax.fill_between(xs, sm, alpha=0.1, color=color)
            if row == 0:
                ax.set_title(title, fontsize=8, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("Steps", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle("PPO Training Curves — Synthetic LOBEnv", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_comparison(results, path):
    """Bar chart comparison of all agents across key metrics."""
    agents = list(results.keys())
    metrics = ["pnl", "sharpe", "mdd", "cvar", "inv", "fill_bid"]
    labels  = ["Net P&L", "Sharpe", "Max Drawdown", "CVaR 95%", "Mean |Inv|", "Fill Rate (Bid)"]
    colors  = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#FF5722"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    fig.suptitle("Agent Comparison — Synthetic LOBEnv (mean ± std)",
                 fontsize=12, fontweight="bold")

    bar_colors = ["#3F51B5", "#FF9800", "#4CAF50", "#E91E63"][:len(agents)]

    for ax, metric, label in zip(axes, metrics, labels):
        means = [results[a][metric].mean() for a in agents]
        stds  = [results[a][metric].std()  for a in agents]
        bars  = ax.bar(range(len(agents)), means, yerr=stds,
                       color=bar_colors, capsize=5, alpha=0.85)
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, rotation=25, ha="right", fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + abs(b.get_height()) * 0.03,
                    f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_tables(results, out_dir):
    """Save LaTeX and CSV tables for the paper."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Method":     name,
            "PnL":        f"{r['pnl'].mean():.3f}",
            "PnL_std":    f"{r['pnl'].std():.3f}",
            "Sharpe":     f"{r['sharpe'].mean():.3f}",
            "Sortino":    f"{r['sortino'].mean():.3f}",
            "MaxDD":      f"{r['mdd'].mean():.3f}",
            "CVaR95":     f"{r['cvar'].mean():.3f}",
            "Mean|Inv|":  f"{r['inv'].mean():.3f}",
            "BidFill":    f"{r['fill_bid'].mean():.3f}",
        })
    df = pd.DataFrame(rows)

    # CSV
    csv_path = os.path.join(out_dir, "sim_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # LaTeX
    tex_path = os.path.join(out_dir, "sim_results.tex")
    caption = ("Performance on synthetic LOBEnv simulation (mean over 20 episodes). "
               "Best value per column in \\textbf{bold}.")
    header = ("\\begin{table}[H]\n\\centering\n"
              f"\\caption{{{caption}}}\n"
              "\\label{tab:sim_results}\n"
              "\\small\n"
              "\\begin{tabular}{@{}lrrrrrrrr@{}}\n"
              "\\toprule\n"
              "Method & P\\&L & Std & Sharpe & Sortino & MaxDD & CVaR95 & Mean|Inv| & BidFill \\\\\n"
              "\\midrule\n")
    body = ""
    for _, row in df.iterrows():
        body += (f"{row['Method']} & {row['PnL']} & {row['PnL_std']} & "
                 f"{row['Sharpe']} & {row['Sortino']} & {row['MaxDD']} & "
                 f"{row['CVaR95']} & {row['Mean|Inv|']} & {row['BidFill']} \\\\\n")
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open(tex_path, "w") as f:
        f.write(header + body + footer)
    print(f"  Saved: {tex_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    backbone = AvellanedaStoikov(**AS_PARAMS)

    # 1. Train Pure RL (model_free)
    pure_cfg = replace(BASE_CFG, policy_mode="model_free")
    pure_trainer, pure_history = train_agent(
        pure_cfg, backbone=None, label="Pure RL", ckpt_prefix="sim_pure_rl")

    # 2. Train Hybrid AS+RL
    hybrid_cfg = replace(BASE_CFG, policy_mode="hybrid")
    hybrid_trainer, hybrid_history = train_agent(
        hybrid_cfg, backbone=backbone, label="Hybrid AS+RL", ckpt_prefix="sim_hybrid")

    # 3. Plot training curves
    plot_training_curves(
        {"Pure RL": pure_history, "Hybrid AS+RL": hybrid_history},
        os.path.join(OUTPUT_DIR, "sim_training_curves.png"))

    # 4. Evaluate all agents
    agents = {
        "Static AS":      StaticASAgent(**AS_PARAMS),
        "Constrained AS": ConstrainedASAgent(AS_PARAMS, max_inv=30, clip_frac=0.8),
        "Pure RL":        pure_trainer,
        "Hybrid AS+RL":   hybrid_trainer,
    }
    results = evaluate_all(agents, n_episodes=20)

    # 5. Comparison figure
    plot_comparison(results, os.path.join(OUTPUT_DIR, "sim_comparison.png"))

    # 6. Save tables
    save_tables(results, OUTPUT_DIR)

    print(f"\n  All synthetic outputs saved to {OUTPUT_DIR}/")
