from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

PPO = Any

from as_backbone import ASBackbone
from config import CFG, Config
from hybrid_env import make_hybrid_env, make_hybrid_tuning_env, make_model_free_env
from mbt_factory import make_trading_env
from mbt_gym.agents.BaselineAgents import AvellanedaStoikovAgent


@dataclass
class EpisodeRecord:
    pnl_trace: np.ndarray = field(default_factory=lambda: np.zeros(1))
    inv_trace: np.ndarray = field(default_factory=lambda: np.zeros(1))
    reward_trace: np.ndarray = field(default_factory=lambda: np.zeros(1))
    fills_trace: np.ndarray = field(default_factory=lambda: np.zeros(1))
    spread_trace: np.ndarray = field(default_factory=lambda: np.zeros(1))


@dataclass
class AgentMetrics:
    mean_pnl: float = 0.0
    std_pnl: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    cvar_5: float = 0.0
    mean_abs_inv: float = 0.0
    std_inv: float = 0.0
    inv_tail_mass: float = 0.0
    time_at_limit: float = 0.0
    fill_rate: float = 0.0
    adv_sel_mean: float = 0.0
    mean_spread: float = 0.0
    std_spread: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    calmar: float = 0.0
    pnl_q05: float = 0.0
    pnl_q25: float = 0.0
    pnl_median: float = 0.0
    pnl_q75: float = 0.0
    pnl_q95: float = 0.0
    ret_skew: float = 0.0
    ret_kurtosis_excess: float = 0.0
    downside_dev: float = 0.0
    episode_len_mean: float = 0.0
    episode_len_std: float = 0.0
    fills_per_episode: float = 0.0
    inv_mean: float = 0.0
    mean_pnl_se: float = 0.0
    mean_pnl_ci_low: float = 0.0
    mean_pnl_ci_high: float = 0.0
    n_episodes: int = 0
    pnl_episodes: List[np.ndarray] = field(default_factory=list)
    inv_episodes: List[np.ndarray] = field(default_factory=list)


def _run_episode(env, action_fn: Callable[[np.ndarray], np.ndarray]) -> EpisodeRecord:
    obs, _ = env.reset()
    done = False

    pnl, inv, rew, fills, spr = [], [], [], [], []

    while not done:
        action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        pnl.append(float(info.get("pnl", 0.0)))
        inv.append(float(info.get("inventory", 0.0)))
        rew.append(float(reward))
        fills.append(float(info.get("n_fills", 0.0)))

        if "hybrid_bid_depth" in info and "hybrid_ask_depth" in info:
            spr.append(float(info["hybrid_bid_depth"] + info["hybrid_ask_depth"]))
        elif "as_spread" in info:
            spr.append(float(info["as_spread"]))
        else:
            # model-free action = [bid_depth, ask_depth]
            a = np.asarray(action).reshape(-1)
            spr.append(float(a[0] + a[1]) if len(a) >= 2 else 0.0)

    env.close()
    return EpisodeRecord(
        pnl_trace=np.asarray(pnl, dtype=float),
        inv_trace=np.asarray(inv, dtype=float),
        reward_trace=np.asarray(rew, dtype=float),
        fills_trace=np.asarray(fills, dtype=float),
        spread_trace=np.asarray(spr, dtype=float),
    )


def _run_static_as(cfg: Config, seed: int, regime: str = "standard") -> EpisodeRecord:
    base = make_trading_env(cfg=cfg, seed=seed, regime=regime)
    agent = AvellanedaStoikovAgent(env=base, risk_aversion=cfg.as_cfg.risk_aversion)
    obs = base.reset()
    done = False

    pnl, inv, rew, fills, spr = [], [], [], [], []
    wealth0 = float(base.state[0, 0] + base.state[0, 1] * base.state[0, 3])
    prev_inv = float(base.state[0, 1])

    while not done:
        action2d = agent.get_action(base.state)
        obs, reward, dones, _infos = base.step(action2d)
        done = bool(np.asarray(dones).reshape(-1)[0])

        s = base.state[0]
        cash, inventory, mid = float(s[0]), float(s[1]), float(s[3])
        pnl.append(cash + inventory * mid - wealth0)
        inv.append(inventory)
        rew.append(float(np.asarray(reward).reshape(-1)[0]))
        fills.append(abs(inventory - prev_inv))
        prev_inv = inventory
        spr.append(float(action2d[0, 0] + action2d[0, 1]))

    base.close()
    return EpisodeRecord(
        pnl_trace=np.asarray(pnl, dtype=float),
        inv_trace=np.asarray(inv, dtype=float),
        reward_trace=np.asarray(rew, dtype=float),
        fills_trace=np.asarray(fills, dtype=float),
        spread_trace=np.asarray(spr, dtype=float),
    )


def _run_constrained_as(cfg: Config, seed: int, regime: str = "standard") -> EpisodeRecord:
    env = make_model_free_env(cfg=cfg, seed=seed, regime=regime)
    backbone = ASBackbone(
        risk_aversion=cfg.as_cfg.risk_aversion,
        fill_exponent=cfg.fill.fill_exponent,
        volatility=cfg.mid.volatility,
        terminal_time=cfg.mid.terminal_time,
        tick_size=0.01,
    )
    cap = 0.8 * cfg.env.max_inventory

    def policy(_obs: np.ndarray):
        s = env.base.state[0]
        inv = float(np.clip(s[1], -cap, cap))
        tau = max(cfg.mid.terminal_time - float(s[2]), 1e-8)
        b, a = backbone.depths(inv, tau)
        return np.array([b, a], dtype=np.float32)

    return _run_episode(env, policy)


def _run_rl_on_model(model: PPO, env, deterministic: bool = True) -> EpisodeRecord:
    def policy(obs: np.ndarray):
        a, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(a, dtype=np.float32)

    return _run_episode(env, policy)


def compute_metrics(records: List[EpisodeRecord], cfg: Optional[Config] = None) -> AgentMetrics:
    cfg = cfg or CFG
    if not records:
        return AgentMetrics()

    i_max = cfg.env.max_inventory
    terminals = np.array([r.pnl_trace[-1] for r in records if len(r.pnl_trace) > 0], dtype=float)
    if len(terminals) == 0:
        return AgentMetrics()

    m = AgentMetrics()
    m.mean_pnl = float(np.mean(terminals))
    m.std_pnl = float(np.std(terminals))
    m.n_episodes = int(len(terminals))
    m.mean_pnl_se = float(np.std(terminals, ddof=1) / np.sqrt(len(terminals))) if len(terminals) > 1 else 0.0
    if len(terminals) > 1:
        ci_half = 1.96 * m.mean_pnl_se
        m.mean_pnl_ci_low = float(m.mean_pnl - ci_half)
        m.mean_pnl_ci_high = float(m.mean_pnl + ci_half)
    else:
        m.mean_pnl_ci_low = m.mean_pnl
        m.mean_pnl_ci_high = m.mean_pnl
    m.pnl_q05 = float(np.quantile(terminals, 0.05))
    m.pnl_q25 = float(np.quantile(terminals, 0.25))
    m.pnl_median = float(np.quantile(terminals, 0.50))
    m.pnl_q75 = float(np.quantile(terminals, 0.75))
    m.pnl_q95 = float(np.quantile(terminals, 0.95))
    m.hit_rate = float(np.mean(terminals > 0.0))

    ep_lens = np.array([len(r.pnl_trace) for r in records], dtype=float)
    m.episode_len_mean = float(np.mean(ep_lens)) if len(ep_lens) else 0.0
    m.episode_len_std = float(np.std(ep_lens)) if len(ep_lens) else 0.0

    rets_list = [np.diff(r.pnl_trace) for r in records if len(r.pnl_trace) > 1]
    if rets_list:
        all_ret = np.concatenate(rets_list)
        m.mean_reward = float(np.mean(all_ret))
        m.std_reward = float(np.std(all_ret))
        if np.std(all_ret) > 1e-10:
            m.sharpe = float(np.mean(all_ret) / np.std(all_ret) * np.sqrt(252.0))
            centered = all_ret - np.mean(all_ret)
            std = np.std(all_ret)
            if std > 1e-12:
                m.ret_skew = float(np.mean((centered / std) ** 3))
                m.ret_kurtosis_excess = float(np.mean((centered / std) ** 4) - 3.0)
        downside = all_ret[all_ret < 0]
        if len(downside) > 0 and np.std(downside) > 1e-10:
            m.sortino = float(np.mean(all_ret) / np.std(downside) * np.sqrt(252.0))
        m.downside_dev = float(np.std(downside)) if len(downside) else 0.0

        neg = -all_ret
        var95 = np.quantile(neg, 0.95)
        tail = neg[neg >= var95]
        m.cvar_5 = float(np.mean(tail)) if len(tail) else 0.0
        pos_sum = float(np.sum(all_ret[all_ret > 0]))
        neg_sum = float(-np.sum(all_ret[all_ret < 0]))
        m.profit_factor = pos_sum / max(neg_sum, 1e-12)

    dds = []
    for r in records:
        if len(r.pnl_trace) < 2:
            continue
        peak = np.maximum.accumulate(r.pnl_trace)
        dds.append(np.max(peak - r.pnl_trace))
    m.max_drawdown = float(np.mean(dds)) if dds else 0.0
    m.calmar = m.mean_pnl / max(m.max_drawdown, 1e-12)

    all_inv = np.concatenate([r.inv_trace for r in records if len(r.inv_trace) > 0])
    if len(all_inv):
        m.inv_mean = float(np.mean(all_inv))
        m.mean_abs_inv = float(np.mean(np.abs(all_inv)))
        m.std_inv = float(np.std(all_inv))
        m.inv_tail_mass = float(np.mean(np.abs(all_inv) > 0.8 * i_max))
        m.time_at_limit = float(np.mean(np.abs(all_inv) >= i_max))

    total_fills = float(sum(np.sum(r.fills_trace) for r in records))
    total_steps = float(sum(len(r.pnl_trace) for r in records))
    m.fill_rate = total_fills / max(total_steps, 1.0)
    m.fills_per_episode = total_fills / max(float(len(records)), 1.0)

    spreads = np.concatenate([r.spread_trace for r in records if len(r.spread_trace) > 0])
    m.mean_spread = float(np.mean(spreads)) if len(spreads) else 0.0
    m.std_spread = float(np.std(spreads)) if len(spreads) else 0.0

    m.pnl_episodes = [r.pnl_trace for r in records]
    m.inv_episodes = [r.inv_trace for r in records]
    return m


def _as_model_list(model_or_models) -> List[PPO]:
    if model_or_models is None:
        return []
    if isinstance(model_or_models, list):
        return [m for m in model_or_models if m is not None]
    return [model_or_models]


def evaluate_all(
    hybrid_model: Optional[PPO] = None,
    model_free_model: Optional[PPO] = None,
    hybrid_tuning_model: Optional[PPO] = None,
    cfg: Optional[Config] = None,
    n_episodes: int = 50,
    base_seed: int = 999,
    regime: str = "standard",
) -> Dict[str, AgentMetrics]:
    cfg = cfg or CFG
    seeds = [base_seed + i * cfg.eval.eval_seed_stride for i in range(n_episodes)]
    out: Dict[str, AgentMetrics] = {}

    print(f"Evaluating ({regime}): Static AS...")
    static_recs = [_run_static_as(cfg, s, regime=regime) for s in tqdm(seeds)]
    out["static_as"] = compute_metrics(static_recs, cfg)

    print(f"Evaluating ({regime}): Constrained AS...")
    constrained_recs = [_run_constrained_as(cfg, s, regime=regime) for s in tqdm(seeds)]
    out["constrained_as"] = compute_metrics(constrained_recs, cfg)

    model_free_models = _as_model_list(model_free_model)
    if model_free_models:
        print(f"Evaluating ({regime}): Model-Free RL...")
        mf_recs: List[EpisodeRecord] = []
        for model_idx, model in enumerate(model_free_models):
            shifted = [s + model_idx * cfg.eval.train_seed_stride for s in seeds]
            mf_recs.extend(
                [_run_rl_on_model(model, make_model_free_env(cfg=cfg, seed=s, regime=regime)) for s in tqdm(shifted)]
            )
        out["model_free_rl"] = compute_metrics(mf_recs, cfg)

    hybrid_tuning_models = _as_model_list(hybrid_tuning_model)
    if hybrid_tuning_models:
        print(f"Evaluating ({regime}): Hybrid Tuning RL...")
        ht_recs: List[EpisodeRecord] = []
        for model_idx, model in enumerate(hybrid_tuning_models):
            shifted = [s + model_idx * cfg.eval.train_seed_stride for s in seeds]
            ht_recs.extend(
                [_run_rl_on_model(model, make_hybrid_tuning_env(cfg=cfg, seed=s, regime=regime)) for s in tqdm(shifted)]
            )
        out["hybrid_tuning"] = compute_metrics(ht_recs, cfg)

    hybrid_models = _as_model_list(hybrid_model)
    if hybrid_models:
        print(f"Evaluating ({regime}): Hybrid AS+RL...")
        hyb_recs: List[EpisodeRecord] = []
        for model_idx, model in enumerate(hybrid_models):
            shifted = [s + model_idx * cfg.eval.train_seed_stride for s in seeds]
            hyb_recs.extend(
                [_run_rl_on_model(model, make_hybrid_env(cfg=cfg, seed=s, regime=regime)) for s in tqdm(shifted)]
            )
        out["hybrid_as_rl"] = compute_metrics(hyb_recs, cfg)

    return out


def evaluate_across_regimes(
    hybrid_model: Optional[PPO] = None,
    model_free_model: Optional[PPO] = None,
    hybrid_tuning_model: Optional[PPO] = None,
    cfg: Optional[Config] = None,
    n_episodes: int = 50,
    regimes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, AgentMetrics]]:
    cfg = cfg or CFG
    regime_list = list(regimes or cfg.eval.regimes)
    return {
        regime: evaluate_all(
            hybrid_model=hybrid_model,
            model_free_model=model_free_model,
            hybrid_tuning_model=hybrid_tuning_model,
            cfg=cfg,
            n_episodes=n_episodes,
            base_seed=cfg.eval.base_seed,
            regime=regime,
        )
        for regime in regime_list
    }


def build_ablation_configs(cfg: Optional[Config] = None) -> Dict[str, Config]:
    cfg = deepcopy(cfg or CFG)
    variants: Dict[str, Config] = {}

    base = deepcopy(cfg)
    variants["full_hybrid"] = base

    no_cmdp = deepcopy(cfg)
    no_cmdp.ablation.use_cmdp = False
    variants["no_cmdp"] = no_cmdp

    no_shield = deepcopy(cfg)
    no_shield.ablation.use_inventory_shield = False
    variants["no_inventory_shield"] = no_shield

    fixed_sigma = deepcopy(cfg)
    fixed_sigma.ablation.use_adaptive_vol = False
    variants["fixed_sigma_backbone"] = fixed_sigma

    delta_only = deepcopy(cfg)
    delta_only.ablation.use_residual_kappa = False
    variants["delta_only_residual"] = delta_only

    kappa_only = deepcopy(cfg)
    kappa_only.ablation.use_residual_delta = False
    variants["kappa_only_residual"] = kappa_only

    tighter_cmdp = deepcopy(cfg)
    tighter_cmdp.cmdp.inventory_mean_abs_max = min(cfg.cmdp.inventory_mean_abs_max, 1.0)
    tighter_cmdp.cmdp.inventory_var_max = min(cfg.cmdp.inventory_var_max, 4.0)
    tighter_cmdp.cmdp.drawdown_max = min(cfg.cmdp.drawdown_max, 2.0)
    tighter_cmdp.cmdp.cvar_epsilon = min(cfg.cmdp.cvar_epsilon, 1.0)
    variants["tight_constraints"] = tighter_cmdp
    return variants


_NAME_MAP = {
    "static_as": "Static AS",
    "constrained_as": "Constrained AS (Guéant)",
    "model_free_rl": "Model-Free RL (PPO)",
    "hybrid_tuning": "Hybrid Tuning RL",
    "hybrid_as_rl": "Hybrid AS+RL (Proposed)",
}


def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    if std < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - float(np.mean(x))) / std


def _score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Higher is better.
    z_pnl = _zscore(out["mean_pnl"].to_numpy(dtype=float))
    z_sharpe = _zscore(out["sharpe"].to_numpy(dtype=float))
    z_sortino = _zscore(out["sortino"].to_numpy(dtype=float))
    z_fill = _zscore(out["fill_rate"].to_numpy(dtype=float))
    # Lower is better.
    z_dd = _zscore(out["max_drawdown"].to_numpy(dtype=float))
    z_cvar = _zscore(out["cvar_5"].to_numpy(dtype=float))
    z_inv = _zscore(out["mean_abs_inv"].to_numpy(dtype=float))
    z_tail = _zscore(out["inv_tail_mass"].to_numpy(dtype=float))
    z_limit = _zscore(out["time_at_limit"].to_numpy(dtype=float))

    score = (
        0.30 * z_pnl
        + 0.20 * z_sharpe
        + 0.10 * z_sortino
        + 0.05 * z_fill
        - 0.15 * z_dd
        - 0.10 * z_cvar
        - 0.05 * z_inv
        - 0.03 * z_tail
        - 0.02 * z_limit
    )
    out["composite_score"] = score
    out["rank"] = out["composite_score"].rank(ascending=False, method="min").astype(int)
    return out


def _with_static_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base = out[out["method"] == "static_as"]
    base_pnl = float(base["mean_pnl"].iloc[0]) if len(base) else np.nan
    base_sharpe = float(base["sharpe"].iloc[0]) if len(base) else np.nan
    base_dd = float(base["max_drawdown"].iloc[0]) if len(base) else np.nan
    base_cvar = float(base["cvar_5"].iloc[0]) if len(base) else np.nan
    base_inv = float(base["mean_abs_inv"].iloc[0]) if len(base) else np.nan
    out["delta_pnl_vs_static"] = out["mean_pnl"] - base_pnl
    out["delta_sharpe_vs_static"] = out["sharpe"] - base_sharpe
    out["dd_improvement_vs_static"] = base_dd - out["max_drawdown"]
    out["cvar_improvement_vs_static"] = base_cvar - out["cvar_5"]
    out["inventory_improvement_vs_static"] = base_inv - out["mean_abs_inv"]
    return out


def to_scored_overview_table_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    df = _with_static_deltas(_score_and_rank(to_numeric_df(results).copy()))
    cols = [
        "label",
        "mean_pnl",
        "delta_pnl_vs_static",
        "sharpe",
        "delta_sharpe_vs_static",
        "max_drawdown",
        "cvar_5",
        "mean_abs_inv",
        "fill_rate",
    ]
    return df[cols + ["rank"]].sort_values(["rank", "mean_pnl"], ascending=[True, False]).drop(columns=["rank"]).reset_index(drop=True)


def to_detailed_table_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    df = _with_static_deltas(to_numeric_df(results).copy())
    cols = [
        "label",
        "n_episodes",
        "mean_pnl",
        "mean_pnl_ci_low",
        "mean_pnl_ci_high",
        "std_pnl",
        "sharpe",
        "sortino",
        "max_drawdown",
        "cvar_5",
        "mean_abs_inv",
        "std_inv",
        "inv_tail_mass",
        "time_at_limit",
        "fill_rate",
        "mean_spread",
        "std_spread",
        "hit_rate",
        "profit_factor",
        "calmar",
        "pnl_q05",
        "pnl_median",
        "pnl_q95",
        "delta_pnl_vs_static",
        "delta_sharpe_vs_static",
    ]
    return df[cols].sort_values(["mean_pnl"], ascending=[False]).reset_index(drop=True)


def to_table_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    detailed = _with_static_deltas(_score_and_rank(to_numeric_df(results).copy()))
    detailed = detailed.sort_values(["rank", "mean_pnl"], ascending=[True, False]).reset_index(drop=True)
    rows = []
    for _, r in detailed.iterrows():
        rows.append(
            {
                "Method": r["label"],
                "N": int(r["n_episodes"]),
                "Mean PnL": f"{r['mean_pnl']:.4f}",
                "95% CI": f"[{r['mean_pnl_ci_low']:.4f}, {r['mean_pnl_ci_high']:.4f}]",
                "Delta PnL(vs Static)": f"{r['delta_pnl_vs_static']:+.4f}",
                "Sharpe": f"{r['sharpe']:.3f}",
                "Delta Sharpe(vs Static)": f"{r['delta_sharpe_vs_static']:+.3f}",
                "Sortino": f"{r['sortino']:.3f}",
                "Max DD": f"{r['max_drawdown']:.4f}",
                "CVaR 5%": f"{r['cvar_5']:.4f}",
                "Mean |Inv|": f"{r['mean_abs_inv']:.3f}",
                "Inv Tail": f"{r['inv_tail_mass']:.2%}",
                "Time@Limit": f"{r['time_at_limit']:.2%}",
                "Fill Rate": f"{r['fill_rate']:.4f}",
                "Mean Spread": f"{r['mean_spread']:.4f}",
            }
        )
    return pd.DataFrame(rows)


def to_numeric_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    rows = []
    for k, m in results.items():
        rows.append(
            {
                "method": k,
                "label": _NAME_MAP.get(k, k),
                "mean_pnl": m.mean_pnl,
                "std_pnl": m.std_pnl,
                "sharpe": m.sharpe,
                "sortino": m.sortino,
                "max_drawdown": m.max_drawdown,
                "cvar_5": m.cvar_5,
                "mean_abs_inv": m.mean_abs_inv,
                "std_inv": m.std_inv,
                "inv_tail_mass": m.inv_tail_mass,
                "time_at_limit": m.time_at_limit,
                "fill_rate": m.fill_rate,
                "mean_spread": m.mean_spread,
                "std_spread": m.std_spread,
                "mean_reward": m.mean_reward,
                "std_reward": m.std_reward,
                "hit_rate": m.hit_rate,
                "profit_factor": m.profit_factor,
                "calmar": m.calmar,
                "pnl_q05": m.pnl_q05,
                "pnl_q25": m.pnl_q25,
                "pnl_median": m.pnl_median,
                "pnl_q75": m.pnl_q75,
                "pnl_q95": m.pnl_q95,
                "ret_skew": m.ret_skew,
                "ret_kurtosis_excess": m.ret_kurtosis_excess,
                "downside_dev": m.downside_dev,
                "episode_len_mean": m.episode_len_mean,
                "episode_len_std": m.episode_len_std,
                "fills_per_episode": m.fills_per_episode,
                "inv_mean": m.inv_mean,
                "mean_pnl_se": m.mean_pnl_se,
                "mean_pnl_ci_low": m.mean_pnl_ci_low,
                "mean_pnl_ci_high": m.mean_pnl_ci_high,
                "n_episodes": m.n_episodes,
            }
        )
    return pd.DataFrame(rows)


def to_comprehensive_table_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    df = _with_static_deltas(to_numeric_df(results).copy())
    ordered = [
        "method",
        "label",
        "n_episodes",
        "mean_pnl",
        "mean_pnl_se",
        "mean_pnl_ci_low",
        "mean_pnl_ci_high",
        "std_pnl",
        "pnl_q05",
        "pnl_q25",
        "pnl_median",
        "pnl_q75",
        "pnl_q95",
        "hit_rate",
        "delta_pnl_vs_static",
        "sharpe",
        "sortino",
        "calmar",
        "delta_sharpe_vs_static",
        "mean_reward",
        "std_reward",
        "downside_dev",
        "ret_skew",
        "ret_kurtosis_excess",
        "max_drawdown",
        "cvar_5",
        "dd_improvement_vs_static",
        "inv_mean",
        "mean_abs_inv",
        "std_inv",
        "inv_tail_mass",
        "time_at_limit",
        "fill_rate",
        "fills_per_episode",
        "mean_spread",
        "std_spread",
        "profit_factor",
        "episode_len_mean",
        "episode_len_std",
    ]
    return df[ordered].sort_values(["mean_pnl"], ascending=[False]).reset_index(drop=True)


def to_returns_table_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    df = _with_static_deltas(to_numeric_df(results).copy())
    cols = [
        "label",
        "mean_pnl",
        "std_pnl",
        "pnl_q05",
        "pnl_q25",
        "pnl_median",
        "pnl_q75",
        "pnl_q95",
        "hit_rate",
        "profit_factor",
        "delta_pnl_vs_static",
    ]
    return df[cols].sort_values(["mean_pnl"], ascending=[False]).reset_index(drop=True)


def to_risk_table_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    df = _with_static_deltas(to_numeric_df(results).copy())
    cols = [
        "label",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "cvar_5",
        "downside_dev",
        "ret_skew",
        "ret_kurtosis_excess",
        "delta_sharpe_vs_static",
        "dd_improvement_vs_static",
        "cvar_improvement_vs_static",
    ]
    return df[cols].sort_values(["sharpe"], ascending=[False]).reset_index(drop=True)


def to_inventory_execution_table_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    df = _with_static_deltas(to_numeric_df(results).copy())
    cols = [
        "label",
        "inv_mean",
        "mean_abs_inv",
        "std_inv",
        "inv_tail_mass",
        "time_at_limit",
        "fill_rate",
        "fills_per_episode",
        "mean_spread",
        "std_spread",
        "inventory_improvement_vs_static",
    ]
    return df[cols].sort_values(["mean_abs_inv"], ascending=[True]).reset_index(drop=True)


# ── NEW TABLE 1: Statistical significance (t-test vs Static AS baseline) ──────

def to_statistical_significance_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    """
    Welch two-sample t-test of terminal PnL for each method against Static AS.
    Reports: t-statistic, p-value, significance stars, effect size (Cohen's d).
    """
    try:
        from scipy import stats as scipy_stats
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    def _terminal_pnls(key: str) -> np.ndarray:
        eps = results[key].pnl_episodes
        vals = [float(np.asarray(ep).ravel()[-1]) for ep in eps if len(ep) > 0]
        return np.asarray(vals, dtype=float) if vals else np.zeros(1)

    baseline_key = "static_as"
    base_pnl = _terminal_pnls(baseline_key)

    rows = []
    for key, m in results.items():
        pnl = _terminal_pnls(key)
        n_self = len(pnl)
        delta_mean = float(np.mean(pnl) - np.mean(base_pnl))

        # pooled std for Cohen's d
        s_pool = float(np.sqrt((np.std(pnl, ddof=1) ** 2 + np.std(base_pnl, ddof=1) ** 2) / 2))
        cohens_d = delta_mean / s_pool if s_pool > 1e-12 else 0.0

        if key == baseline_key or not _has_scipy:
            t_stat, p_val = float("nan"), float("nan")
            stars = "—"
        else:
            res = scipy_stats.ttest_ind(pnl, base_pnl, equal_var=False)
            t_stat = float(res.statistic)
            p_val  = float(res.pvalue)
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            else:
                stars = "n.s."

        rows.append({
            "Method":            _NAME_MAP.get(key, key),
            "N episodes":        n_self,
            "Mean PnL":          f"{float(np.mean(pnl)):.4f}",
            "Delta vs Static":   f"{delta_mean:+.4f}",
            "t-statistic":       f"{t_stat:.3f}" if not np.isnan(t_stat) else "—",
            "p-value":           f"{p_val:.4f}" if not np.isnan(p_val) else "—",
            "Significance":      stars,
            "Cohen's d":         f"{cohens_d:.3f}",
        })

    return pd.DataFrame(rows)


# ── NEW TABLE 2: Risk-adjusted performance ────────────────────────────────────

def to_risk_adjusted_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    """
    Sharpe, Sortino, Calmar, Omega ratio, Hit Rate, Profit Factor,
    downside deviation, and Ret skewness — sorted by Sharpe.
    """
    rows = []
    for key, m in results.items():
        # Omega ratio: E[max(r,0)] / E[max(-r,0)]
        eps = results[key].pnl_episodes
        all_rets: List[float] = []
        for ep in eps:
            a = np.asarray(ep, dtype=float).ravel()
            if len(a) > 1:
                all_rets.extend(np.diff(a).tolist())
        rets = np.asarray(all_rets, dtype=float) if all_rets else np.zeros(1)
        gains  = float(np.mean(np.maximum(rets, 0)))
        losses = float(np.mean(np.maximum(-rets, 0)))
        omega  = gains / max(losses, 1e-12)

        rows.append({
            "Method":           _NAME_MAP.get(key, key),
            "Sharpe":           f"{m.sharpe:.3f}",
            "Sortino":          f"{m.sortino:.3f}",
            "Calmar":           f"{m.calmar:.3f}",
            "Omega Ratio":      f"{omega:.3f}",
            "Hit Rate":         f"{m.hit_rate:.2%}",
            "Profit Factor":    f"{m.profit_factor:.3f}",
            "Downside Dev":     f"{m.downside_dev:.4f}",
            "Return Skewness":  f"{m.ret_skew:.3f}",
            "Excess Kurtosis":  f"{m.ret_kurtosis_excess:.3f}",
        })

    df = pd.DataFrame(rows)
    df["_sharpe_sort"] = [float(results[k].sharpe) for k in results]
    return df.sort_values("_sharpe_sort", ascending=False).drop(columns=["_sharpe_sort"]).reset_index(drop=True)


# ── NEW TABLE 3: Full PnL distribution moments ────────────────────────────────

def to_pnl_distribution_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    """
    Complete distributional statistics for terminal PnL:
    mean, std, skewness, excess kurtosis, min, Q5, Q25, median, Q75, Q95, max.
    """
    rows = []
    for key, m in results.items():
        eps = results[key].pnl_episodes
        terminals = np.asarray(
            [float(np.asarray(ep).ravel()[-1]) for ep in eps if len(ep) > 0],
            dtype=float,
        )
        if len(terminals) == 0:
            terminals = np.zeros(1)

        skew = float(
            np.mean(((terminals - terminals.mean()) / max(terminals.std(), 1e-12)) ** 3)
        )
        kurt = float(
            np.mean(((terminals - terminals.mean()) / max(terminals.std(), 1e-12)) ** 4) - 3
        )

        rows.append({
            "Method":           _NAME_MAP.get(key, key),
            "N":                len(terminals),
            "Mean":             f"{float(np.mean(terminals)):.4f}",
            "Std":              f"{float(np.std(terminals)):.4f}",
            "Skewness":         f"{skew:.3f}",
            "Excess Kurtosis":  f"{kurt:.3f}",
            "Min":              f"{float(np.min(terminals)):.4f}",
            "Q5":               f"{float(np.quantile(terminals, 0.05)):.4f}",
            "Q25":              f"{float(np.quantile(terminals, 0.25)):.4f}",
            "Median":           f"{float(np.quantile(terminals, 0.50)):.4f}",
            "Q75":              f"{float(np.quantile(terminals, 0.75)):.4f}",
            "Q95":              f"{float(np.quantile(terminals, 0.95)):.4f}",
            "Max":              f"{float(np.max(terminals)):.4f}",
        })

    df = pd.DataFrame(rows)
    df["_sort"] = [float(np.mean([float(np.asarray(ep).ravel()[-1])
                                  for ep in results[k].pnl_episodes if len(ep) > 0] or [0]))
                   for k in results]
    return df.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)


# ── NEW TABLE 4: Per-metric rank table ────────────────────────────────────────

def to_per_metric_rank_df(results: Dict[str, AgentMetrics]) -> pd.DataFrame:
    """
    Rank of each method on each key metric (1 = best).
    Higher-is-better metrics ranked descending; risk metrics ranked ascending.
    """
    higher_better = ["mean_pnl", "sharpe", "sortino", "calmar",
                     "hit_rate", "profit_factor", "fill_rate"]
    lower_better  = ["max_drawdown", "cvar_5", "mean_abs_inv",
                     "std_inv", "inv_tail_mass", "time_at_limit", "downside_dev"]

    metric_display = {
        "mean_pnl":      "Mean PnL",
        "sharpe":        "Sharpe",
        "sortino":       "Sortino",
        "calmar":        "Calmar",
        "hit_rate":      "Hit Rate",
        "profit_factor": "Profit Factor",
        "fill_rate":     "Fill Rate",
        "max_drawdown":  "Max DD",
        "cvar_5":        "CVaR 5%",
        "mean_abs_inv":  "Mean |Inv|",
        "std_inv":       "Std Inv",
        "inv_tail_mass": "Inv Tail",
        "time_at_limit": "Time@Limit",
        "downside_dev":  "Downside Dev",
    }

    all_metrics = higher_better + lower_better
    dfn = to_numeric_df(results).copy()

    rank_df = pd.DataFrame()
    rank_df["Method"] = dfn["label"].values

    for col in all_metrics:
        if col not in dfn.columns:
            continue
        vals = dfn[col].to_numpy(float)
        asc  = col in lower_better
        # rank: 1 = best
        order = np.argsort(vals if asc else -vals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        rank_df[metric_display[col]] = ranks

    # overall average rank
    rank_cols = [metric_display[c] for c in all_metrics if c in dfn.columns]
    rank_df["Avg Rank"] = rank_df[rank_cols].mean(axis=1).round(2)
    return rank_df.sort_values("Avg Rank").reset_index(drop=True)


# ── NEW TABLE 5: Constraint satisfaction summary ──────────────────────────────

def to_constraint_satisfaction_df(
    results: Dict[str, AgentMetrics],
    drawdown_bound: float = 5.0,
    cvar_bound: float = 10.0,
) -> pd.DataFrame:
    """
    Fraction of episodes where max-drawdown and CVaR-5% stay within
    the CMDP constraint bounds used during training.
    Also reports mean constraint slack (bound - realised value).
    """
    rows = []
    for key, m in results.items():
        eps = results[key].pnl_episodes

        # per-episode max drawdown
        ep_dd = []
        for ep in eps:
            pnl = np.asarray(ep, dtype=float).ravel()
            if len(pnl) < 2:
                ep_dd.append(0.0)
                continue
            peak = np.maximum.accumulate(pnl)
            ep_dd.append(float(np.max(peak - pnl)))
        ep_dd_arr = np.asarray(ep_dd, dtype=float) if ep_dd else np.zeros(1)

        # per-episode CVaR-5% (on step-returns)
        ep_cvar = []
        for ep in eps:
            pnl = np.asarray(ep, dtype=float).ravel()
            if len(pnl) < 10:
                ep_cvar.append(0.0)
                continue
            neg_rets = -np.diff(pnl)
            var95 = np.quantile(neg_rets, 0.95)
            tail  = neg_rets[neg_rets >= var95]
            ep_cvar.append(float(np.mean(tail)) if len(tail) else 0.0)
        ep_cvar_arr = np.asarray(ep_cvar, dtype=float) if ep_cvar else np.zeros(1)

        dd_sat   = float(np.mean(ep_dd_arr   <= drawdown_bound))
        cvar_sat = float(np.mean(ep_cvar_arr <= cvar_bound))

        rows.append({
            "Method":                   _NAME_MAP.get(key, key),
            "DD bound":                 f"{drawdown_bound:.1f}",
            "Mean MaxDD":               f"{float(np.mean(ep_dd_arr)):.4f}",
            "DD satisfied (%)":         f"{dd_sat:.1%}",
            "DD slack (mean)":          f"{drawdown_bound - float(np.mean(ep_dd_arr)):+.4f}",
            "CVaR bound":               f"{cvar_bound:.1f}",
            "Mean CVaR 5%":             f"{float(np.mean(ep_cvar_arr)):.4f}",
            "CVaR satisfied (%)":       f"{cvar_sat:.1%}",
            "CVaR slack (mean)":        f"{cvar_bound - float(np.mean(ep_cvar_arr)):+.4f}",
            "Both satisfied (%)":       f"{float(np.mean((ep_dd_arr <= drawdown_bound) & (ep_cvar_arr <= cvar_bound))):.1%}",
        })

    return pd.DataFrame(rows)


def to_regime_summary_df(regime_results: Dict[str, Dict[str, AgentMetrics]]) -> pd.DataFrame:
    rows = []
    for regime, results in regime_results.items():
        for key, metrics in results.items():
            rows.append(
                {
                    "regime": regime,
                    "method": key,
                    "label": _NAME_MAP.get(key, key),
                    "n_episodes": metrics.n_episodes,
                    "mean_pnl": metrics.mean_pnl,
                    "mean_pnl_ci_low": metrics.mean_pnl_ci_low,
                    "mean_pnl_ci_high": metrics.mean_pnl_ci_high,
                    "sharpe": metrics.sharpe,
                    "max_drawdown": metrics.max_drawdown,
                    "cvar_5": metrics.cvar_5,
                    "mean_abs_inv": metrics.mean_abs_inv,
                    "fill_rate": metrics.fill_rate,
                }
            )
    return pd.DataFrame(rows).sort_values(["regime", "mean_pnl"], ascending=[True, False]).reset_index(drop=True)


def to_ablation_summary_df(ablation_results: Dict[str, Dict[str, Dict[str, AgentMetrics]]]) -> pd.DataFrame:
    rows = []
    for ablation_name, regime_pack in ablation_results.items():
        for regime, results in regime_pack.items():
            if "hybrid_as_rl" not in results:
                continue
            m = results["hybrid_as_rl"]
            rows.append(
                {
                    "ablation": ablation_name,
                    "regime": regime,
                    "n_episodes": m.n_episodes,
                    "mean_pnl": m.mean_pnl,
                    "mean_pnl_ci_low": m.mean_pnl_ci_low,
                    "mean_pnl_ci_high": m.mean_pnl_ci_high,
                    "sharpe": m.sharpe,
                    "max_drawdown": m.max_drawdown,
                    "cvar_5": m.cvar_5,
                    "mean_abs_inv": m.mean_abs_inv,
                }
            )
    return pd.DataFrame(rows).sort_values(["regime", "mean_pnl"], ascending=[True, False]).reset_index(drop=True)
