"""
评估指标 + 基线策略
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    net_pnl:        float = 0.0
    sharpe:         float = 0.0
    sortino:        float = 0.0
    max_drawdown:   float = 0.0
    cvar_95:        float = 0.0
    mean_inventory: float = 0.0
    std_inventory:  float = 0.0
    max_abs_inv:    float = 0.0
    time_at_limit:  float = 0.0
    fill_rate_bid:  float = 0.0
    fill_rate_ask:  float = 0.0
    mean_adv_sel:   float = 0.0
    n_steps:        int   = 0


def compute_metrics(pnl, inv, bid_f, ask_f, adv, max_inv=50.0):
    m = EpisodeMetrics()
    m.n_steps   = len(pnl)
    cum         = np.cumsum(pnl)
    m.net_pnl   = float(cum[-1]) if len(cum) else 0.0

    if pnl.std() > 1e-10:
        m.sharpe = float(pnl.mean() / pnl.std() * np.sqrt(len(pnl)))

    # Sortino: 下行标准差为 0 时若均值 > 0 则为 +inf（全正收益，风险为零）
    dn = pnl[pnl < 0]
    if len(dn) > 0 and dn.std() > 1e-10:
        m.sortino = float(pnl.mean() / dn.std() * np.sqrt(len(pnl)))
    elif pnl.mean() > 1e-10:
        m.sortino = float("inf")
    # else: 均值 <= 0 且无下行波动，sortino 保持 0.0

    peak        = np.maximum.accumulate(cum)
    m.max_drawdown = float((peak - cum).max()) if len(cum) else 0.0

    # CVaR-95%: 取最差 5% 的平均损失，用 ceil 确保覆盖完整分位
    n_tail = max(int(np.ceil(0.05 * len(pnl))), 1)
    m.cvar_95 = float(-np.sort(pnl)[:n_tail].mean()) if len(pnl) else 0.0

    m.mean_inventory = float(np.mean(np.abs(inv)))
    m.std_inventory  = float(np.std(inv))
    m.max_abs_inv    = float(np.max(np.abs(inv)))
    m.time_at_limit  = float(np.mean(np.abs(inv) >= max_inv * 0.9))

    m.fill_rate_bid = float(np.mean(bid_f)) if len(bid_f) else 0.0
    m.fill_rate_ask = float(np.mean(ask_f)) if len(ask_f) else 0.0
    m.mean_adv_sel  = float(np.mean(adv))   if len(adv)   else 0.0
    return m


def print_metrics(m: EpisodeMetrics, label=""):
    print(f"\n{'─'*48}")
    if label:
        print(f"  {label}")
    print(f"{'─'*48}")
    print(f"  Net P&L      : {m.net_pnl:>12.4f}")
    print(f"  Sharpe       : {m.sharpe:>12.4f}")
    print(f"  Sortino      : {m.sortino:>12.4f}")
    print(f"  Max Drawdown : {m.max_drawdown:>12.4f}")
    print(f"  CVaR 95%     : {m.cvar_95:>12.4f}")
    print(f"  Mean |Inv|   : {m.mean_inventory:>12.4f}")
    print(f"  Max  |Inv|   : {m.max_abs_inv:>12.4f}")
    print(f"  Time@Limit   : {m.time_at_limit:>12.4f}")
    print(f"  Fill Bid     : {m.fill_rate_bid:>12.4f}")
    print(f"  Fill Ask     : {m.fill_rate_ask:>12.4f}")
    print(f"  Adv.Sel.     : {m.mean_adv_sel:>12.6f}")
    print(f"{'─'*48}")


# ── 基线策略 ──────────────────────────────────────────────────

class StaticASAgent:
    """静态 AS 基线"""
    def __init__(self, gamma=0.1, sigma=0.02, k=1.5, A=5.0,
                 horizon=0.5, tick_size=0.01):
        from models.avellaneda_stoikov import AvellanedaStoikov
        self.m = AvellanedaStoikov(gamma, sigma, k, A, horizon, tick_size)

    def act(self, obs, mid, inventory, t_rem, sigma_hat=None, **kw):
        bid, ask, _, _ = self.m.quotes(mid, inventory, t_rem, sigma_hat)
        bo = max(mid - bid, self.m.tick_size)
        ao = max(ask - mid, self.m.tick_size)
        return np.array([np.log(bo), np.log(ao), 0.0, 0.0], dtype=np.float32)


class SymmetricAgent:
    """固定对称价差基线"""
    def __init__(self, half_spread=2, tick_size=0.01):
        self.hs = half_spread * tick_size

    def act(self, obs, mid, inventory, t_rem, sigma_hat=None, **kw):
        return np.array([np.log(self.hs), np.log(self.hs), 0., 0.], dtype=np.float32)


class RandomAgent:
    """随机基线（下界）"""
    def __init__(self, tick_size=0.01, seed=99):
        self.ts  = tick_size
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mid, inventory, t_rem, sigma_hat=None, **kw):
        bo = self.rng.uniform(1, 5) * self.ts
        ao = self.rng.uniform(1, 5) * self.ts
        return np.array([np.log(bo), np.log(ao), 0., 0.], dtype=np.float32)
