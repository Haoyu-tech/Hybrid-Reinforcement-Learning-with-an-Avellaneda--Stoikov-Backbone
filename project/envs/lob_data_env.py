"""
LOBDataEnv — 数据回放做市商环境
=================================
用真实 OHLCV 分钟数据替代 GBM 仿真：
- 价格序列来自历史数据（BTC/ETH 永续合约）
- 填单模型：OHLC crossing（分钟内低价<=bid 则成交）
- 价格在 episode 内归一化到 S0=100，保证策略跨价格段泛化
- 观测维度与 LOBEnv 保持一致（12 维），可直接复用 PPO agent
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StepInfo:
    bid_filled:        bool  = False
    ask_filled:        bool  = False
    fill_bid_price:    float = 0.0
    fill_ask_price:    float = 0.0
    adverse_selection: float = 0.0
    fee_cost:          float = 0.0
    cancel_cost:       float = 0.0
    inventory_penalty: float = 0.0
    terminal_cost:     float = 0.0


class LOBDataEnv:
    """
    数据驱动的做市商仿真环境。

    动作空间: [log_bid_offset, log_ask_offset, log_bid_size, log_ask_size]
    观测空间: 12 维归一化向量

    填单规则（OHLC crossing）:
        bid_price = mid - exp(action[0])
        bid 成交 ⟺ bar_low  <= bid_price
        ask_price = mid + exp(action[1])
        ask 成交 ⟺ bar_high >= ask_price

    价格归一化:
        每个 episode 开始时将价格归一化至 S0_norm=100，
        使同一策略可适用于 BTC 从 $4k 到 $100k 的不同价位。
    """

    OBS_DIM    = 12
    S0_NORM    = 100.0       # 归一化基准价格

    def __init__(
        self,
        df: pd.DataFrame,            # OHLCV 数据，含 open/high/low/close/volume 列
        episode_steps:    int   = 500,
        tick_size:        float = 0.01,   # 归一化空间中的最小价格单位
        max_inventory:    float = 10.0,
        initial_cash:     float = 1e6,
        fee_rate:         float = 2e-4,   # Binance 永续合约 maker 费率
        cancel_cost:      float = 1e-5,
        lambda_inventory: float = 0.005,
        lambda_adverse:   float = 0.3,
        terminal_penalty: float = 0.10,
        flatten_fee_rate: float = 7e-4,
        shield_start_frac: float = 0.60,
        shield_intensity: float = 1.25,
        vol_window:       int   = 20,     # 滚动波动率窗口（分钟）
        vol_thresholds:   Optional[Tuple[float, float]] = None,  # regime 阈值
        seed:             int   = 42,
    ):
        # 数据预处理
        self._raw = df[['open','high','low','close','volume']].reset_index(drop=True).astype(float)
        self._n   = len(self._raw)

        self.episode_steps    = episode_steps
        self.tick_size        = tick_size
        self.max_inventory    = max_inventory
        self.initial_cash     = initial_cash
        self.fee_rate         = fee_rate
        self.cancel_cost      = cancel_cost
        self.lambda_inventory = lambda_inventory
        self.lambda_adverse   = lambda_adverse
        self.terminal_penalty = terminal_penalty
        self.flatten_fee_rate = flatten_fee_rate
        self.shield_start_frac = shield_start_frac
        self.shield_intensity = shield_intensity
        self.vol_window       = vol_window

        # 预计算整列 mid-price 和收益率用于特征提取
        self._mid_all = (self._raw['high'] + self._raw['low']).values / 2.0
        self._ret_all = pd.Series(self._mid_all).pct_change().fillna(0).values
        self._vol_all = (pd.Series(self._ret_all)
                         .rolling(vol_window, min_periods=1).std()
                         .fillna(0).values)
        self._vol_mean = float(np.nanmean(self._vol_all[self._vol_all > 0]) or 1e-6)

        # regime 阈值：从数据计算，或由外部传入（用训练集计算后传给测试集）
        if vol_thresholds is None:
            nz = self._vol_all[self._vol_all > 0]
            self._vol_lo = float(np.percentile(nz, 33))
            self._vol_hi = float(np.percentile(nz, 67))
        else:
            self._vol_lo, self._vol_hi = vol_thresholds

        # 预计算相对成交量（proxy for 流动性）
        self._vol_ratio = (pd.Series(self._raw['volume'].values)
                           .rolling(vol_window, min_periods=1).mean()
                           .fillna(1).values)

        self.sigma_base = self._vol_mean   # 归一化参考波动率

        self.rng = np.random.default_rng(seed)
        self._ep_start = 0
        self._reset_state(0)

    # ── Gym 接口 ──────────────────────────────────────────────

    def reset(self, seed=None, start_idx: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if start_idx is not None:
            idx = int(np.clip(start_idx, 0,
                              self._n - self.episode_steps - 2))
        else:
            max_start = self._n - self.episode_steps - 2
            idx = int(self.rng.integers(max(0, max_start)))
        self._reset_state(idx)
        return self._get_obs(), {}

    def step(self, action):
        row   = self._raw.iloc[self.step_idx + self._ep_start]
        high_n = row['high'] / self._S0_real * self.S0_NORM
        low_n  = row['low']  / self._S0_real * self.S0_NORM
        mid_n  = (high_n + low_n) / 2.0

        bid_off = max(self._tick(np.exp(float(action[0]))), self.tick_size)
        ask_off = max(self._tick(np.exp(float(action[1]))), self.tick_size)
        bid_sz  = float(np.clip(np.exp(float(action[2])), 0.1, 5.0))
        ask_sz  = float(np.clip(np.exp(float(action[3])), 0.1, 5.0))
        bid_off, ask_off, bid_sz, ask_sz = self._inventory_shield(bid_off, ask_off, bid_sz, ask_sz)

        bid_price = self._tick(mid_n - bid_off)
        ask_price = self._tick(mid_n + ask_off)
        bid_price = min(bid_price, mid_n - self.tick_size)
        ask_price = max(ask_price, mid_n + self.tick_size)

        mid_before = self.mid

        # 更新到下一根 bar
        self.step_idx += 1
        next_row  = self._raw.iloc[self.step_idx + self._ep_start]
        nh = next_row['high'] / self._S0_real * self.S0_NORM
        nl = next_row['low']  / self._S0_real * self.S0_NORM
        self.mid = (nh + nl) / 2.0
        self._update_features(self.step_idx + self._ep_start)
        mid_after = self.mid

        # OHLC crossing 填单判断
        info = StepInfo()
        bid_filled = bool(low_n  <= bid_price)
        ask_filled = bool(high_n >= ask_price)
        info.bid_filled = bid_filled
        info.ask_filled = ask_filled

        buy_capacity = max(self.max_inventory - self.inventory, 0.0)
        sell_capacity = max(self.max_inventory + self.inventory, 0.0)
        bid_exec = min(bid_sz, buy_capacity)
        ask_exec = min(ask_sz, sell_capacity)

        if bid_filled and bid_exec > 0.0:
            self.inventory += bid_exec
            self.cash      -= bid_price * bid_exec
            fee             = self.fee_rate * bid_price * bid_exec
            self.cash      -= fee
            info.fee_cost  += fee
            info.fill_bid_price = bid_price
        else:
            info.bid_filled = False

        if ask_filled and ask_exec > 0.0:
            self.inventory -= ask_exec
            self.cash      += ask_price * ask_exec
            fee             = self.fee_rate * ask_price * ask_exec
            self.cash      -= fee
            info.fee_cost  += fee
            info.fill_ask_price = ask_price
        else:
            info.ask_filled = False

        cancel = self.cancel_cost * (1 + int(info.bid_filled) + int(info.ask_filled))
        self.cash     -= cancel
        info.cancel_cost = cancel

        # 逆向选择成本（价格在成交后不利方向运动）
        adv = 0.0
        if info.bid_filled:
            adv += max(0.0, mid_before - mid_after) * bid_exec
        if info.ask_filled:
            adv += max(0.0, mid_after - mid_before) * ask_exec
        info.adverse_selection = adv

        portfolio      = self.cash + self.inventory * self.mid
        delta_pnl      = portfolio - self._prev_portfolio

        t_rem = (self.episode_steps - self.step_idx) / self.episode_steps
        late_factor = 1.0 + 3.0 * (1.0 - t_rem) ** 2
        inv_pen        = self.lambda_inventory * late_factor * (self.inventory ** 2)
        adv_pen        = self.lambda_adverse   * adv
        info.inventory_penalty = inv_pen

        terminal_cost = 0.0
        truncated  = (self.step_idx >= self.episode_steps)
        if truncated and abs(self.inventory) > 1e-8:
            flatten_notional = abs(self.inventory) * self.mid
            terminal_cost = self.flatten_fee_rate * flatten_notional + self.terminal_penalty * abs(self.inventory)
            self.cash -= terminal_cost
            self.cash += self.inventory * self.mid
            self.inventory = 0.0
            portfolio = self.cash
            delta_pnl = portfolio - self._prev_portfolio
        self._prev_portfolio = portfolio
        info.terminal_cost = terminal_cost

        reward = delta_pnl - info.fee_cost - cancel - inv_pen - adv_pen - terminal_cost

        terminated = False
        return self._get_obs(), float(reward), terminated, truncated, info

    # ── 内部方法 ──────────────────────────────────────────────

    def _reset_state(self, start_idx: int):
        self._ep_start = start_idx
        self._S0_real  = float(self._mid_all[start_idx])   # 真实起始价格
        self.mid       = self.S0_NORM                       # 归一化起始 mid
        self.inventory = 0.0
        self.cash      = self.initial_cash
        self.step_idx  = 0
        self._prev_portfolio = self.initial_cash
        self._update_features(start_idx)

    def _update_features(self, abs_idx: int):
        """更新当前 bar 的派生特征"""
        self.sigma_hat = float(self._vol_all[abs_idx])
        self._regime   = self._get_regime(abs_idx)
        self._vol_r    = float(
            self._raw['volume'].iloc[abs_idx] /
            (self._vol_ratio[abs_idx] + 1e-8))
        row = self._raw.iloc[abs_idx]
        self._bar_range = float(
            (row['high'] - row['low']) / self._S0_real * self.S0_NORM)
        denom = (row['high'] - row['low'] + 1e-8)
        self._imbalance = float((row['close'] - row['open']) / denom)

    def _get_regime(self, abs_idx: int) -> int:
        v = self._vol_all[abs_idx]
        if v < self._vol_lo:  return 0
        if v < self._vol_hi:  return 1
        return 2

    def _get_obs(self) -> np.ndarray:
        abs_idx = self.step_idx + self._ep_start
        t_rem   = (self.episode_steps - self.step_idx) / self.episode_steps
        mid_dev = (self.mid - self.S0_NORM) / self.S0_NORM
        obs = np.array([
            mid_dev,                                                  # 0: 价格偏移
            self.inventory / self.max_inventory,                      # 1: 归一化库存
            self._bar_range / (self.tick_size + 1e-8) / 100.0,       # 2: bar range (ticks)
            float(np.clip(self._imbalance, -1, 1)),                   # 3: 方向压力
            (self.mid - self.S0_NORM) / (self.tick_size + 1e-8) / 1000.0,  # 4: mid偏移(ticks)
            float(self.sigma_hat / (self.sigma_base + 1e-8)),         # 5: 波动率比
            t_rem,                                                     # 6: 剩余时间
            self._regime / 2.0,                                       # 7: regime
            (self.cash - self.initial_cash) / 1000.0,                 # 8: 累计现金PnL
            (self.inventory ** 2) / (self.max_inventory ** 2),        # 9: 平方库存
            float(np.clip(self._vol_r, 0, 5) / 5.0),                 # 10: 相对成交量
            float(np.clip(self._bar_range / (self.S0_NORM * 0.01), 0, 5) / 5.0),  # 11: 归一化range
        ], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _tick(self, v: float) -> float:
        return round(v / self.tick_size) * self.tick_size

    def _inventory_shield(self, bid_off: float, ask_off: float, bid_sz: float, ask_sz: float):
        inv_frac = float(np.clip(self.inventory / (self.max_inventory + 1e-8), -1.0, 1.0))
        excess = max(abs(inv_frac) - self.shield_start_frac, 0.0)
        if excess <= 0.0:
            return bid_off, ask_off, bid_sz, ask_sz

        strength = 1.0 + self.shield_intensity * excess / max(1.0 - self.shield_start_frac, 1e-8)
        if inv_frac > 0.0:
            bid_off *= strength
            ask_off = max(ask_off / strength, self.tick_size)
            bid_sz *= max(0.15, 1.0 / (strength ** 2))
            ask_sz *= min(strength, 2.0)
        else:
            ask_off *= strength
            bid_off = max(bid_off / strength, self.tick_size)
            ask_sz *= max(0.15, 1.0 / (strength ** 2))
            bid_sz *= min(strength, 2.0)
        return bid_off, ask_off, bid_sz, ask_sz

    @property
    def portfolio_value(self) -> float:
        return self.cash + self.inventory * self.mid


# ── 向量化包装器 ─────────────────────────────────────────────

class VecLOBDataEnv:
    """n_envs 个 LOBDataEnv 并行，接口与 VecLOBEnv 一致"""

    def __init__(self, n_envs: int, df: pd.DataFrame, env_kwargs: dict,
                 vol_thresholds=None):
        if vol_thresholds is not None:
            env_kwargs = {**env_kwargs, 'vol_thresholds': vol_thresholds}
        self.n_envs  = n_envs
        self.envs    = [LOBDataEnv(df=df, seed=i * 100 + 1, **env_kwargs)
                        for i in range(n_envs)]
        self.obs_dim = LOBDataEnv.OBS_DIM

    def reset(self, seed=None):
        obs_list = []
        for i, env in enumerate(self.envs):
            s = (seed + i) if seed is not None else None
            o, _ = env.reset(seed=s)
            obs_list.append(o)
        return np.stack(obs_list)

    def step(self, actions):
        obs_l, rew_l, done_l, info_l = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(actions[i])
            done = term or trunc
            if done:
                o, _ = env.reset()
            obs_l.append(o)
            rew_l.append(r)
            done_l.append(done)
            info_l.append(info)
        return (np.stack(obs_l),
                np.array(rew_l, dtype=np.float32),
                np.array(done_l, dtype=bool),
                info_l)
