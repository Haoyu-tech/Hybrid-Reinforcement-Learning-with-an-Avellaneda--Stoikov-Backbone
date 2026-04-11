"""
整理实证结果，生成学术图表，输出到 results/ 文件夹
运行: python make_figures.py
"""

import os, sys, shutil
from dataclasses import replace
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

from train_data import (
    BASE_PPO_CFG,
    DATA_PATH,
    ENV_KWARGS,
    TEST_PERIODS,
    TRAIN_PERIOD,
    StaticASAgent,
    load_period,
)
from envs.lob_data_env import LOBDataEnv
from agents.ppo import PPOTrainer
from models.avellaneda_stoikov import AvellanedaStoikov

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams


# ── 全局绘图风格（学术论文风格）────────────────────────────────
rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

SRC  = os.path.join(_HERE, "outputs")
DEST = os.path.join(_HERE, "results")
os.makedirs(DEST, exist_ok=True)

AGENT_COLORS = {
    "Static AS":    "#FF9800",
    "Pure RL":      "#4CAF50",
    "Hybrid AS+RL": "#2196F3",
}
AGENT_MARKERS = {"Static AS": "s", "Pure RL": "o", "Hybrid AS+RL": "^"}
AGENTS = ["Static AS", "Pure RL", "Hybrid AS+RL"]

PERIOD_LABELS = {
    "COVID_Crash_2020Q1":    "COVID\nCrash\n2020Q1",
    "FTX_Collapse_2022Q4":   "FTX\nCollapse\n2022Q4",
    "Bull_Run_2023Q1":       "Bull Run\n2023Q1",
    "LowVol_Sideways_2023Q3":"LowVol\n2023Q3",
    "ETF_Rally_2024Q1":      "ETF Rally\n2024Q1",
    "ATH_2024Q4":            "ATH\n2024Q4",
}

# ── 读取数据 ─────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(SRC, "data_period_results.csv"))
periods = list(df["period"].unique())
n_periods = len(periods)


def _load_btc_context():
    df_full = pd.read_csv(DATA_PATH)
    df_full["dt"] = pd.to_datetime(df_full["timestamp"], unit="ms")
    df_train = load_period(df_full, *TRAIN_PERIOD)

    mid_train = (df_train["high"] + df_train["low"]).values / 2.0
    ret_train = pd.Series(mid_train).pct_change().dropna().values
    sigma_eff = float(ret_train.std()) * np.sqrt(ENV_KWARGS["episode_steps"])

    tmp_env = LOBDataEnv(df=df_train, seed=0, **ENV_KWARGS)
    vol_thresholds = (tmp_env._vol_lo, tmp_env._vol_hi)

    summary_path = os.path.join(SRC, "as_calibration_summary.csv")
    summary = pd.read_csv(summary_path).set_index("item")["value"].to_dict()
    as_params = {
        "gamma": float(summary["best_gamma"]),
        "k": float(summary["best_k"]),
        "A": float(summary["A_fit_from_fill_curve"]),
        "horizon": float(summary["best_horizon"]),
        "sigma": float(summary.get("sigma_eff", sigma_eff)),
        "tick_size": ENV_KWARGS["tick_size"],
    }
    return df_full, sigma_eff, vol_thresholds, as_params


def _load_agents(as_params):
    pure_cfg = replace(BASE_PPO_CFG, policy_mode="model_free")
    hybrid_cfg = replace(BASE_PPO_CFG, policy_mode="hybrid")

    pure = PPOTrainer(pure_cfg, backbone=None)
    pure.load(os.path.join(SRC, "data_pure_rl_final.pt"))

    hybrid_backbone = AvellanedaStoikov(**as_params)
    hybrid = PPOTrainer(hybrid_cfg, backbone=hybrid_backbone)
    hybrid.load(os.path.join(SRC, "data_hybrid_final.pt"))

    agents = {
        "Static AS": StaticASAgent(as_params),
        "Pure RL": pure,
        "Hybrid AS+RL": hybrid,
    }
    return agents


def _act(agent, obs, env, sigma_eff):
    mid = env.mid
    inv_now = env.inventory
    t_rem = (env.episode_steps - env.step_idx) / env.episode_steps
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
        return decoded.squeeze(0).astype(np.float32)
    return np.asarray(agent.act(obs, mid, inv_now, t_rem, sig), dtype=np.float32)


def _collect_episode_trace(agent, env, start_idx, sigma_eff):
    obs, _ = env.reset(start_idx=int(start_idx))
    pnl_path = [0.0]
    done = False
    while not done:
        act = _act(agent, obs, env, sigma_eff)
        obs, _, term, trunc, _info = env.step(act)
        pnl_path.append(float(env.portfolio_value - env.initial_cash))
        done = term or trunc
    return np.asarray(pnl_path, dtype=float)


def _collect_empirical_paths(selected_periods=None, n_episodes=20):
    df_full, sigma_eff, vol_thresholds, as_params = _load_btc_context()
    agents = _load_agents(as_params)
    selected_periods = selected_periods or list(TEST_PERIODS.keys())
    data = {}

    for period_name in selected_periods:
        start, end = TEST_PERIODS[period_name]
        df_period = load_period(df_full, start, end)
        env = LOBDataEnv(df=df_period, seed=0, vol_thresholds=vol_thresholds, **ENV_KWARGS)
        env.sigma_base = sigma_eff
        max_start = len(df_period) - ENV_KWARGS["episode_steps"] - 2
        starts = np.linspace(0, max_start, n_episodes, dtype=int)

        period_data = {}
        for agent_name, agent in agents.items():
            traces = [_collect_episode_trace(agent, env, start_idx, sigma_eff) for start_idx in starts]
            arr = np.vstack(traces)
            period_data[agent_name] = {
                "paths": arr,
                "mean": arr.mean(axis=0),
                "std": arr.std(axis=0),
                "terminal": arr[:, -1],
            }
        data[period_name] = period_data
    return data


# ════════════════════════════════════════════════════════════════
# Fig 1: PnL 跨区间对比（主图，论文核心图）
# ════════════════════════════════════════════════════════════════
def fig_pnl_comparison():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(n_periods)
    w = 0.25
    for i, agent in enumerate(AGENTS):
        sub = df[df["agent"] == agent].set_index("period")
        vals  = [sub.loc[p, "pnl_mean"] if p in sub.index else 0.0 for p in periods]
        stds  = [sub.loc[p, "pnl_std"]  if p in sub.index else 0.0 for p in periods]
        bars = ax.bar(x + (i - 1) * w, vals, w, yerr=stds,
                      color=AGENT_COLORS[agent], alpha=0.85,
                      capsize=4, label=agent, zorder=3)
        for bar, v in zip(bars, vals):
            yoff = max(abs(v) * 0.04, 0.5)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (yoff if v >= 0 else -yoff * 3),
                    f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7.5, fontweight="bold",
                    color=AGENT_COLORS[agent])

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in periods], fontsize=9)
    ax.set_ylabel("Net P&L (normalized price units)", fontsize=10)
    ax.set_title("Out-of-Sample Net P&L Across Six Market Regimes\n"
                 "(BTC USDT Perpetual, 1-min OHLCV, 20 episodes per period)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(DEST, "fig1_pnl_comparison.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {path}")


# ════════════════════════════════════════════════════════════════
# Fig 2: Sharpe × MaxDD 散点图（风险-收益权衡）
# ════════════════════════════════════════════════════════════════
def fig_risk_return():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 左: Sharpe 跨区间
    ax = axes[0]
    x = np.arange(n_periods)
    w = 0.25
    for i, agent in enumerate(AGENTS):
        sub = df[df["agent"] == agent].set_index("period")
        vals = [sub.loc[p, "sharpe"] if p in sub.index else 0.0 for p in periods]
        ax.bar(x + (i - 1) * w, vals, w,
               color=AGENT_COLORS[agent], alpha=0.85, label=agent, zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in periods], fontsize=8.5)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio by Market Regime", fontweight="bold")
    ax.legend(fontsize=8.5)

    # 右: MaxDD 跨区间
    ax = axes[1]
    for i, agent in enumerate(AGENTS):
        sub = df[df["agent"] == agent].set_index("period")
        vals = [sub.loc[p, "max_dd"] if p in sub.index else 0.0 for p in periods]
        ax.bar(x + (i - 1) * w, vals, w,
               color=AGENT_COLORS[agent], alpha=0.85, label=agent, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in periods], fontsize=8.5)
    ax.set_ylabel("Maximum Drawdown")
    ax.set_title("Max Drawdown by Market Regime", fontweight="bold")
    ax.legend(fontsize=8.5)

    plt.suptitle("Risk Metrics Across Market Regimes", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(DEST, "fig2_risk_metrics.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {path}")


# ════════════════════════════════════════════════════════════════
# Fig 3: Fill Rate × Inventory（做市行为分析）
# ════════════════════════════════════════════════════════════════
def fig_market_making_behavior():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    x = np.arange(n_periods)
    w = 0.25
    for i, agent in enumerate(AGENTS):
        sub = df[df["agent"] == agent].set_index("period")
        fill = [0.5 * (sub.loc[p, "fill_bid"] + sub.loc[p, "fill_ask"])
                if p in sub.index else 0.0 for p in periods]
        ax.bar(x + (i - 1) * w, fill, w,
               color=AGENT_COLORS[agent], alpha=0.85, label=agent, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in periods], fontsize=8.5)
    ax.set_ylabel("Fill Rate (bid+ask avg)")
    ax.set_title("Average Fill Rate by Market Regime", fontweight="bold")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    for i, agent in enumerate(AGENTS):
        sub = df[df["agent"] == agent].set_index("period")
        inv = [sub.loc[p, "inv_mean"] if p in sub.index else 0.0 for p in periods]
        ax.bar(x + (i - 1) * w, inv, w,
               color=AGENT_COLORS[agent], alpha=0.85, label=agent, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in periods], fontsize=8.5)
    ax.set_ylabel("Mean |Inventory|")
    ax.set_title("Mean Absolute Inventory by Market Regime", fontweight="bold")
    ax.legend(fontsize=8.5)

    plt.suptitle("Market-Making Behavior Analysis", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(DEST, "fig3_behavior.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {path}")


# ════════════════════════════════════════════════════════════════
# Fig 4: 综合雷达图（6区间 × 3策略 × PnL）
# ════════════════════════════════════════════════════════════════
def fig_radar():
    cats = [PERIOD_LABELS[p].replace("\n", " ") for p in periods]
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=9)

    for agent in AGENTS:
        sub = df[df["agent"] == agent].set_index("period")
        vals = [max(sub.loc[p, "pnl_mean"], 0) if p in sub.index else 0.0 for p in periods]
        # 归一化到 [0,1]
        max_v = max(vals) if max(vals) > 0 else 1
        vals_n = [v / max_v for v in vals] + [vals[0] / max_v]
        ax.plot(angles, vals_n, "o-", lw=2, color=AGENT_COLORS[agent], label=agent)
        ax.fill(angles, vals_n, alpha=0.10, color=AGENT_COLORS[agent])

    ax.set_title("Relative P&L Performance\nAcross Market Regimes (normalized per-period)",
                 fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    path = os.path.join(DEST, "fig4_radar.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {path}")


# ════════════════════════════════════════════════════════════════
# Fig 5: AS 填单曲线校准图
# ════════════════════════════════════════════════════════════════
def fig_fill_curve():
    fc_path = os.path.join(SRC, "as_fill_curve.csv")
    if not os.path.exists(fc_path):
        print("  [skip] as_fill_curve.csv not found")
        return
    fc = pd.read_csv(fc_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.scatter(fc["offset"], fc["fill_side"], color="#2196F3", zorder=5,
               s=60, label="Empirical fill rate")
    if "fill_fit" in fc.columns:
        ax.plot(fc["offset"], fc["fill_fit"], "r--", lw=1.8,
                label=r"Fitted: $1-e^{-Ae^{-k\delta}}$")
    ax.set_xlabel("Half-spread offset δ (price units, S₀=100)")
    ax.set_ylabel("Fill probability P(fill)")
    ax.set_title("OHLC Fill Curve Calibration\n(BTC 1m Training Data, 2020-06-01~2022-09-30)",
                 fontweight="bold")
    ax.legend()

    ax = axes[1]
    if "intensity" in fc.columns and "log_intensity_fit" in fc.columns:
        fit_df = fc[fc["intensity"] > 1e-6]
        ax.scatter(fit_df["offset"], np.log(fit_df["intensity"]),
                   color="#FF9800", zorder=5, s=60, label="log λ(δ) empirical")
        ax.plot(fc["offset"], fc["log_intensity_fit"], "r--", lw=1.8,
                label=r"Linear fit: $\log A - k\delta$")
        ax.set_xlabel("Half-spread offset δ")
        ax.set_ylabel("log Fill Intensity log λ(δ)")
        ax.set_title("Exponential Intensity Fit\n(MLE for Avellaneda-Stoikov k parameter)",
                     fontweight="bold")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(DEST, "fig5_fill_curve_calibration.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {path}")


# ════════════════════════════════════════════════════════════════
# Fig 6: 训练曲线（Pure RL + Hybrid，合并展示）
# ════════════════════════════════════════════════════════════════
def fig_training_curves():
    # 直接复制已有图（高质量重绘需要 history，这里用已存文件）
    src_pure   = os.path.join(SRC, "data_pure_rl_training_curves.png")
    src_hybrid = os.path.join(SRC, "data_hybrid_training_curves.png")
    if not os.path.exists(src_pure) or not os.path.exists(src_hybrid):
        print("  [skip] training curve pngs not found")
        return

    from PIL import Image
    img_pure   = np.array(Image.open(src_pure))
    img_hybrid = np.array(Image.open(src_hybrid))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].imshow(img_pure);   axes[0].axis("off"); axes[0].set_title("Pure RL Training", fontsize=12, fontweight="bold")
    axes[1].imshow(img_hybrid); axes[1].axis("off"); axes[1].set_title("Hybrid AS+RL Training", fontsize=12, fontweight="bold")
    plt.suptitle("PPO Training Curves — Real BTC 1-min Data (3M steps each)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(DEST, "fig6_training_curves.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved -> {path}")


# ════════════════════════════════════════════════════════════════
# Table: LaTeX 综合汇总表（跨区间平均）
# ════════════════════════════════════════════════════════════════
def fig_btc_equity_curves():
    selected = list(TEST_PERIODS.keys())
    path_data = _collect_empirical_paths(selected_periods=selected, n_episodes=20)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes = axes.ravel()
    steps = np.arange(ENV_KWARGS["episode_steps"] + 1)

    for ax, period_name in zip(axes, selected):
        for agent in AGENTS:
            mean = path_data[period_name][agent]["mean"]
            std = path_data[period_name][agent]["std"]
            ax.plot(steps, mean, lw=2.0, color=AGENT_COLORS[agent], label=agent)
            ax.fill_between(steps, mean - std, mean + std, color=AGENT_COLORS[agent], alpha=0.12)
        ax.axhline(0, color="black", lw=0.8, alpha=0.7)
        ax.set_title(PERIOD_LABELS.get(period_name, period_name).replace("\n", " "), fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Episode step (1 step = 1 minute)")
        ax.grid(True, alpha=0.25)
    for ax in axes[::3]:
        ax.set_ylabel("Cumulative net P&L")
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    plt.suptitle(
        "BTC Empirical Equity Curves Across Six Out-of-Sample Regimes\n(mean +/- 1 s.d. across 20 episodes per regime)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(DEST, "fig11_btc_equity_curves.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {path}")


def fig_btc_terminal_distribution():
    path_data = _collect_empirical_paths(selected_periods=list(TEST_PERIODS.keys()), n_episodes=20)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes = axes.ravel()

    for ax, period_name in zip(axes, TEST_PERIODS.keys()):
        samples = [path_data[period_name][agent]["terminal"] for agent in AGENTS]
        pos = np.arange(1, len(AGENTS) + 1)
        vp = ax.violinplot(samples, positions=pos, widths=0.8, showmeans=False, showmedians=False, showextrema=False)
        for body, agent in zip(vp["bodies"], AGENTS):
            body.set_facecolor(AGENT_COLORS[agent])
            body.set_edgecolor(AGENT_COLORS[agent])
            body.set_alpha(0.20)

        bp = ax.boxplot(samples, positions=pos, widths=0.28, patch_artist=True, showfliers=False)
        for patch, agent in zip(bp["boxes"], AGENTS):
            patch.set_facecolor(AGENT_COLORS[agent])
            patch.set_alpha(0.60)
            patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.4)

        ax.axhline(0, color="black", lw=0.8, alpha=0.7)
        ax.set_xticks(pos)
        ax.set_xticklabels(AGENTS, rotation=18, ha="right", fontsize=8)
        ax.set_title(PERIOD_LABELS.get(period_name, period_name).replace("\n", " "), fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Terminal net P&L")
        ax.grid(True, axis="y", alpha=0.25)

    plt.suptitle(
        "BTC Empirical Terminal P&L Distributions by Market Regime\n(violin + box plot, 20 out-of-sample episodes per regime)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(DEST, "fig12_btc_terminal_distribution.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {path}")


def table_overall_summary():
    # 按 agent 计算跨区间均值
    grp = df.groupby("agent").agg(
        PnL_mean=("pnl_mean", "mean"),
        PnL_std=("pnl_mean", "std"),
        Sharpe=("sharpe", "mean"),
        Sortino=("sortino", "mean"),
        MaxDD=("max_dd", "mean"),
        CVaR95=("cvar95", "mean"),
        MeanInv=("inv_mean", "mean"),
        FillBid=("fill_bid", "mean"),
        FillAsk=("fill_ask", "mean"),
        AdvSel=("adv_sel", "mean"),
    ).reset_index()

    grp["PnL"] = grp.apply(lambda r: f"{r.PnL_mean:.2f} $\\pm$ {r.PnL_std:.2f}", axis=1)
    grp = grp[["agent","PnL","Sharpe","Sortino","MaxDD","CVaR95","MeanInv","FillBid","AdvSel"]]
    grp.columns = ["Method","P\\&L","Sharpe","Sortino","MaxDD","CVaR95","Mean|Inv|","BidFill","AdvSel"]

    # 保存 CSV
    csv_path = os.path.join(DEST, "table1_overall_summary.csv")
    grp.to_csv(csv_path, index=False)

    # 保存 LaTeX
    tex_path = os.path.join(DEST, "table1_overall_summary.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Out-of-Sample Performance Summary (Mean across 6 market regimes, 20 episodes each)}\n")
        f.write("\\label{tab:overall}\n")
        f.write("\\small\n")
        f.write(grp.to_latex(index=False, escape=False,
                              float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))
        f.write("\\end{table}\n")
    print(f"  Saved -> {csv_path}")
    print(f"  Saved -> {tex_path}")


def table_by_period():
    out = df[["period","agent","pnl_mean","pnl_std","sharpe","sortino","max_dd","cvar95","inv_mean","fill_bid","fill_ask","adv_sel"]].copy()
    out["period"] = out["period"].map(lambda p: PERIOD_LABELS.get(p, p).replace("\n", " "))
    out.columns = ["Period","Method","PnL","PnL_std","Sharpe","Sortino","MaxDD","CVaR95","MeanInv","FillBid","FillAsk","AdvSel"]

    csv_path = os.path.join(DEST, "table2_by_period.csv")
    tex_path = os.path.join(DEST, "table2_by_period.tex")
    out.to_csv(csv_path, index=False)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Out-of-Sample Performance by Market Regime}\n")
        f.write("\\label{tab:by_period}\n")
        f.write("\\small\n")
        f.write("\\begin{adjustbox}{max width=\\textwidth}\n")
        f.write(out.drop(columns=["PnL_std"]).to_latex(
            index=False, escape=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))
        f.write("\\end{adjustbox}\n")
        f.write("\\end{table}\n")
    print(f"  Saved -> {csv_path}")
    print(f"  Saved -> {tex_path}")


# ════════════════════════════════════════════════════════════════
# Fig 7: 相对优势图（RL vs Static AS 的 P&L 倍数 + MaxDD 降低幅度）
# ════════════════════════════════════════════════════════════════
def fig_relative_advantage():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(n_periods)
    w = 0.35

    # 左: P&L 相对倍数 vs Static AS
    ax = axes[0]
    for i, agent in enumerate(["Pure RL", "Hybrid AS+RL"]):
        ratios = []
        for p in periods:
            sub = df[df["period"] == p].set_index("agent")
            base = sub.loc["Static AS", "pnl_mean"] if "Static AS" in sub.index else 1.0
            val  = sub.loc[agent, "pnl_mean"]       if agent    in sub.index else 0.0
            ratios.append(val / base if abs(base) > 1e-6 else 0.0)
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, ratios, w,
                      color=AGENT_COLORS[agent], alpha=0.85, label=agent, zorder=3)
        for bar, v in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    f"{v:.1f}×", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=AGENT_COLORS[agent])
    ax.axhline(1.0, color=AGENT_COLORS["Static AS"], lw=1.5, ls="--", label="Static AS (1×)")
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in periods], fontsize=8.5)
    ax.set_ylabel("P&L Multiple Relative to Static AS")
    ax.set_title("Relative P&L Advantage over Static AS\n(×-multiple per market regime)", fontweight="bold")
    ax.legend(fontsize=8.5)

    # 右: MaxDD 降低百分比
    ax = axes[1]
    for i, agent in enumerate(["Pure RL", "Hybrid AS+RL"]):
        reductions = []
        for p in periods:
            sub = df[df["period"] == p].set_index("agent")
            base_dd = sub.loc["Static AS", "max_dd"] if "Static AS" in sub.index else 1.0
            agent_dd = sub.loc[agent, "max_dd"]       if agent    in sub.index else 0.0
            reductions.append((base_dd - agent_dd) / base_dd * 100 if base_dd > 1e-6 else 0.0)
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, reductions, w,
                      color=AGENT_COLORS[agent], alpha=0.85, label=agent, zorder=3)
        for bar, v in zip(bars, reductions):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=AGENT_COLORS[agent])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in periods], fontsize=8.5)
    ax.set_ylabel("Max Drawdown Reduction (%)")
    ax.set_title("Maximum Drawdown Reduction vs Static AS\n(positive = lower drawdown than Static AS)", fontweight="bold")
    ax.legend(fontsize=8.5)

    plt.suptitle("RL Advantage Over Static AS Baseline", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(DEST, "fig7_relative_advantage.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {path}")


# ════════════════════════════════════════════════════════════════
# Fig 8: Sharpe vs MaxDD 散点图（跨区间×策略，Pareto 前沿）
# ════════════════════════════════════════════════════════════════
def fig_sharpe_maxdd_scatter():
    PERIOD_SHORT = {
        "COVID_Crash_2020Q1":    "COVID",
        "FTX_Collapse_2022Q4":   "FTX",
        "Bull_Run_2023Q1":       "Bull",
        "LowVol_Sideways_2023Q3":"LowVol",
        "ETF_Rally_2024Q1":      "ETF",
        "ATH_2024Q4":            "ATH",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for agent in AGENTS:
        sub = df[df["agent"] == agent]
        xs  = sub["max_dd"].values
        ys  = sub["sharpe"].values
        ax.scatter(xs, ys, color=AGENT_COLORS[agent],
                   marker=AGENT_MARKERS[agent], s=90, zorder=5,
                   label=agent, edgecolors="white", lw=0.5)
        for _, row in sub.iterrows():
            ax.annotate(PERIOD_SHORT.get(row["period"], row["period"]),
                        (row["max_dd"], row["sharpe"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7.5, color=AGENT_COLORS[agent], alpha=0.85)

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Maximum Drawdown (normalised price units)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio vs Maximum Drawdown Across Six Market Regimes\n"
                 "(each marker = one agent × regime; upper-left = preferred)",
                 fontweight="bold")
    ax.legend(framealpha=0.9)

    # 添加箭头标注 "preferred region"
    ax.annotate("Preferred\nregion", xy=(0.5, 3.0), fontsize=9,
                color="gray", style="italic",
                arrowprops=dict(arrowstyle="->", color="gray"),
                xytext=(2.0, 4.5))

    plt.tight_layout()
    path = os.path.join(DEST, "fig8_sharpe_maxdd_scatter.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {path}")


# ════════════════════════════════════════════════════════════════
# Fig 9: 尾部风险 + 做市效率分解（Sortino, CVaR95, AdvSel, Fill/PnL）
# ════════════════════════════════════════════════════════════════
def fig_tail_risk_efficiency():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(n_periods)
    w = 0.25

    metrics = [
        ("sortino",  "Sortino Ratio",                 axes[0, 0],
         "Sortino Ratio by Regime\n(downside-only risk adjustment)"),
        ("cvar95",   "CVaR 95% (tail loss per step)", axes[0, 1],
         "CVaR 95% Tail Risk by Regime\n(lower = better tail risk management)"),
        ("adv_sel",  "Adverse Selection Cost",         axes[1, 0],
         "Adverse Selection Cost by Regime\n(P&L lost to informed traders per bar)"),
    ]
    for col, ylabel, ax, title in metrics:
        for i, agent in enumerate(AGENTS):
            sub = df[df["agent"] == agent].set_index("period")
            vals = [sub.loc[p, col] if p in sub.index else 0.0 for p in periods]
            ax.bar(x + (i - 1) * w, vals, w,
                   color=AGENT_COLORS[agent], alpha=0.85, label=agent, zorder=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([PERIOD_LABELS[p] for p in periods], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)

    # 第4图: P&L per unit fill (做市效率)
    ax = axes[1, 1]
    for i, agent in enumerate(AGENTS):
        sub = df[df["agent"] == agent].set_index("period")
        efficiency = []
        for p in periods:
            if p not in sub.index:
                efficiency.append(0.0)
                continue
            avg_fill = 0.5 * (sub.loc[p, "fill_bid"] + sub.loc[p, "fill_ask"])
            pnl = sub.loc[p, "pnl_mean"]
            efficiency.append(pnl / avg_fill if avg_fill > 1e-6 else 0.0)
        ax.bar(x + (i - 1) * w, efficiency, w,
               color=AGENT_COLORS[agent], alpha=0.85, label=agent, zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABELS[p] for p in periods], fontsize=8)
    ax.set_ylabel("P&L / Avg Fill Rate")
    ax.set_title("P&L per Unit Fill Rate by Regime\n(higher = more P&L extracted per trade executed)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)

    plt.suptitle("Tail Risk Decomposition and Market-Making Efficiency",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(DEST, "fig9_tail_risk_efficiency.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {path}")


# ════════════════════════════════════════════════════════════════
# Fig 10: 综合雷达图（多指标 × 3策略跨区间平均）
# ════════════════════════════════════════════════════════════════
def fig_radar_multimetric():
    """Radar chart with 5 metrics averaged across all periods."""
    METRICS = ["P&L", "Sharpe", "Low MaxDD", "Sortino", "Low AdvSel"]
    N = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRICS, fontsize=10)

    # Compute per-agent mean across periods, normalize each metric to [0,1]
    metric_cols = {
        "P&L":         ("pnl_mean",   False),
        "Sharpe":      ("sharpe",     False),
        "Low MaxDD":   ("max_dd",     True),   # invert: lower is better
        "Sortino":     ("sortino",    False),
        "Low AdvSel":  ("adv_sel",    True),   # invert: lower is better
    }
    agent_means = {}
    for agent in AGENTS:
        sub = df[df["agent"] == agent]
        agent_means[agent] = {m: sub[c].mean() for m, (c, _) in metric_cols.items()}

    for metric, (_, invert) in metric_cols.items():
        vals = np.array([agent_means[a][metric] for a in AGENTS])
        vmin, vmax = vals.min(), vals.max()
        span = vmax - vmin if vmax > vmin else 1.0
        for a in AGENTS:
            v = (agent_means[a][metric] - vmin) / span
            agent_means[a][metric + "_norm"] = 1.0 - v if invert else v

    for agent in AGENTS:
        vals_n = [agent_means[agent][m + "_norm"] for m in METRICS]
        vals_n += vals_n[:1]
        ax.plot(angles, vals_n, "o-", lw=2.2, color=AGENT_COLORS[agent], label=agent)
        ax.fill(angles, vals_n, alpha=0.12, color=AGENT_COLORS[agent])

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.5)
    ax.set_title("Multi-Metric Performance Radar\n(averaged across 6 market regimes; outer = better)",
                 fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.40, 1.15), fontsize=9)
    plt.tight_layout()
    path = os.path.join(DEST, "fig10_radar_multimetric.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved -> {path}")


# ════════════════════════════════════════════════════════════════
# 复制模型权重和 calibration 数据
# ════════════════════════════════════════════════════════════════
def copy_assets():
    assets = [
        ("data_pure_rl_final.pt",       "model_pure_rl_final.pt"),
        ("data_hybrid_final.pt",        "model_hybrid_final.pt"),
        ("as_calibration_summary.csv",  "as_calibration_summary.csv"),
        ("as_fill_curve.csv",           "as_fill_curve.csv"),
        ("as_grid_search.csv",          "as_grid_search.csv"),
        ("data_period_results.csv",     "data_period_results.csv"),
    ]
    for src_name, dst_name in assets:
        src = os.path.join(SRC, src_name)
        dst = os.path.join(DEST, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied -> {dst_name}")
        else:
            print(f"  [skip] {src_name} not found")


# ════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Generating academic figures & tables")
    print(f"  Output -> {DEST}")
    print("=" * 60)

    print("\n[Figures]")
    fig_pnl_comparison()
    fig_risk_return()
    fig_market_making_behavior()
    fig_radar()
    fig_fill_curve()
    fig_training_curves()
    fig_btc_equity_curves()
    fig_btc_terminal_distribution()
    fig_relative_advantage()
    fig_sharpe_maxdd_scatter()
    fig_tail_risk_efficiency()
    fig_radar_multimetric()

    print("\n[Tables]")
    table_overall_summary()
    table_by_period()

    print("\n[Assets]")
    copy_assets()

    print("\n" + "=" * 60)
    print(f"  Done. All files in: {DEST}")
    files = sorted(os.listdir(DEST))
    for f in files:
        size = os.path.getsize(os.path.join(DEST, f))
        print(f"    {f:<45s} {size/1024:.1f} KB")
    print("=" * 60)
