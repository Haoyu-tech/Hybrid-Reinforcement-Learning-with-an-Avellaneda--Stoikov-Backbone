"""
LOB 仿真环境（无外部依赖版本）
================================
不依赖 mbt-gym / gymnasium，纯 numpy 实现，
但接口与 mbt-gym 保持一致，方便后续迁移。

特性
----
- GBM + 跳扩散中间价
- Regime switching（正常 / 高波动 / 趋势）
- Poisson 成交模型（AS 风格）
- 完整费用模型：手续费 + 撤单成本 + 逆向选择代理
- 支持向量化（多环境并行）
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class StepInfo:
    bid_filled: bool = False
    ask_filled: bool = False
    fill_bid_price: float = 0.0
    fill_ask_price: float = 0.0
    adverse_selection: float = 0.0
    fee_cost: float = 0.0
    cancel_cost: float = 0.0
    inventory_penalty: float = 0.0


class LOBEnv:
    """
    做市商 LOB 仿真环境。

    动作空间: [log_bid_offset, log_ask_offset, log_bid_size, log_ask_size]
    观测空间: 12 维归一化向量（详见 _get_obs）
    """

    OBS_DIM = 12
    # 动作空间: [log_bid_offset, log_ask_offset, log_bid_size, log_ask_size]
    # 所有维度均为对数空间，env 内部 exp 解码

    def __init__(
        self,
        S0: float = 100.0,
        sigma: float = 0.02,
        mu: float = 0.0,
        jump_prob: float = 0.005,
        jump_scale: float = 0.5,
        tick_size: float = 0.01,
        episode_steps: int = 500,
        dt: float = 1.0,
        max_inventory: int = 50,
        initial_cash: float = 1e6,
        base_intensity: float = 5.0,
        k_flow: float = 1.5,
        fee_rate: float = 2e-4,
        cancel_cost: float = 1e-5,
        lambda_inventory: float = 0.01,
        lambda_adverse: float = 0.3,
        use_regimes: bool = True,
        seed: int = 42,
    ):
        self.S0 = S0
        self.sigma_base = sigma
        self.mu = mu
        self.jump_prob = jump_prob
        self.jump_scale = jump_scale
        self.tick_size = tick_size
        self.episode_steps = episode_steps
        self.dt = dt
        self.max_inventory = max_inventory
        self.initial_cash = initial_cash
        self.base_intensity = base_intensity
        self.k_flow = k_flow
        self.fee_rate = fee_rate
        self.cancel_cost = cancel_cost
        self.lambda_inventory = lambda_inventory
        self.lambda_adverse = lambda_adverse
        self.use_regimes = use_regimes

        # regime 参数: [sigma_mult, intensity_mult, drift_mult]
        self._regime_params = np.array([
            [1.0, 1.0, 0.0],
            [2.5, 0.5, 0.0],
            [1.2, 1.5, 0.3],
        ])
        self._P_regime = np.array([
            [0.97, 0.02, 0.01],
            [0.05, 0.93, 0.02],
            [0.03, 0.02, 0.95],
        ])

        self.rng = np.random.default_rng(seed)
        self._reset_state()

    # ── Gym 接口 ──────────────────────────────────────────────

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action):
        bid_off, ask_off, log_bsz, log_asz = self._parse_action(action)
        bid_price = self._tick(self.mid - bid_off)
        ask_price = self._tick(self.mid + ask_off)
        bid_size  = float(np.clip(np.exp(log_bsz), 0.1, 10.0))
        ask_size  = float(np.clip(np.exp(log_asz), 0.1, 10.0))
        bid_price = min(bid_price, self.mid - self.tick_size)
        ask_price = max(ask_price, self.mid + self.tick_size)

        mid_before = self.mid
        self._evolve_price()
        mid_after  = self.mid

        info = StepInfo()
        bid_filled, ask_filled = self._simulate_fills(
            bid_price, ask_price, bid_size, ask_size, info)

        if bid_filled:
            self.inventory += bid_size
            self.cash -= bid_price * bid_size
            fee = self.fee_rate * bid_price * bid_size
            self.cash -= fee
            info.fee_cost += fee
            info.fill_bid_price = bid_price

        if ask_filled:
            self.inventory -= ask_size
            self.cash += ask_price * ask_size
            fee = self.fee_rate * ask_price * ask_size
            self.cash -= fee
            info.fee_cost += fee
            info.fill_ask_price = ask_price

        cancel = self.cancel_cost * (1 + int(bid_filled) + int(ask_filled))
        self.cash -= cancel
        info.cancel_cost = cancel

        adv = 0.0
        if bid_filled:
            adv += max(0.0, mid_before - mid_after) * bid_size
        if ask_filled:
            adv += max(0.0, mid_after - mid_before) * ask_size
        info.adverse_selection = adv

        self.inventory = float(np.clip(self.inventory,
                                       -self.max_inventory, self.max_inventory))

        portfolio = self.cash + self.inventory * self.mid
        delta_pnl = portfolio - self._prev_portfolio
        self._prev_portfolio = portfolio

        inv_pen = self.lambda_inventory * (self.inventory ** 2)
        adv_pen = self.lambda_adverse * adv
        info.inventory_penalty = inv_pen

        reward = delta_pnl - info.fee_cost - cancel - inv_pen - adv_pen

        self._update_regime()
        self._update_sigma_hat()
        self.step_idx += 1

        terminated = False
        truncated  = self.step_idx >= self.episode_steps
        return self._get_obs(), float(reward), terminated, truncated, info

    # ── 内部辅助 ──────────────────────────────────────────────

    def _reset_state(self):
        self.mid = self.S0
        self.inventory = 0.0
        self.cash = self.initial_cash
        self.step_idx = 0
        self.regime = 0
        self.sigma_hat = self.sigma_base
        self._mid_history = [self.S0]
        self._prev_portfolio = self.initial_cash
        self._bid_depth = 10.0
        self._ask_depth = 10.0

    def _evolve_price(self):
        sp = self._regime_params[self.regime]
        sig   = self.sigma_base * sp[0]
        drift = self.mu + sp[2] * sig
        noise = self.rng.normal(0, sig * self.mid)
        jump  = 0.0
        if self.rng.random() < self.jump_prob:
            jump = self.rng.choice([-1, 1]) * self.jump_scale * self.tick_size * self.mid
        self.mid = max(self.mid + drift + noise + jump, self.tick_size)
        self._mid_history.append(self.mid)
        self._bid_depth = float(self.rng.exponential(10.0))
        self._ask_depth = float(self.rng.exponential(10.0))

    def _simulate_fills(self, bid_price, ask_price, bid_size, ask_size, info):
        sp = self._regime_params[self.regime]
        intensity = self.base_intensity * sp[1]
        lam_b = intensity * np.exp(-self.k_flow * max(self.mid - bid_price, 0))
        lam_a = intensity * np.exp(-self.k_flow * max(ask_price - self.mid, 0))
        bid_filled = self.rng.poisson(lam_b * self.dt) > 0
        ask_filled = self.rng.poisson(lam_a * self.dt) > 0
        info.bid_filled = bid_filled
        info.ask_filled = ask_filled
        return bid_filled, ask_filled

    def _update_regime(self):
        if not self.use_regimes:
            return
        self.regime = int(self.rng.choice(3, p=self._P_regime[self.regime]))

    def _update_sigma_hat(self):
        if len(self._mid_history) >= 20:
            rets = np.diff(np.log(self._mid_history[-20:]))
            self.sigma_hat = float(np.std(rets))
        else:
            self.sigma_hat = self.sigma_base

    def _parse_action(self, action):
        # action = [log_bid_offset, log_ask_offset, log_bid_size, log_ask_size]
        # exp 解码保证偏移量始终为正，与所有 agent 的输出约定一致
        a = np.asarray(action, dtype=float)
        bid_off = max(self._tick(np.exp(float(a[0]))), self.tick_size)
        ask_off = max(self._tick(np.exp(float(a[1]))), self.tick_size)
        return bid_off, ask_off, float(a[2]), float(a[3])

    def _tick(self, v):
        return round(v / self.tick_size) * self.tick_size

    def _get_obs(self):
        total = self._bid_depth + self._ask_depth + 1e-8
        imb   = (self._bid_depth - self._ask_depth) / total
        best_b = self._tick(self.mid - self.tick_size)
        best_a = self._tick(self.mid + self.tick_size)
        micro  = (best_b * self._ask_depth + best_a * self._bid_depth) / total
        t_rem  = (self.episode_steps - self.step_idx) / self.episode_steps
        obs = np.array([
            (self.mid - self.S0) / self.S0,
            self.inventory / self.max_inventory,
            (best_a - best_b) / self.tick_size,
            imb,
            (micro - self.mid) / self.tick_size,
            self.sigma_hat / self.sigma_base,
            t_rem,
            self.regime / 2.0,
            (self.cash - self.initial_cash) / 1000.0,
            (self.inventory ** 2) / (self.max_inventory ** 2),
            self._bid_depth / 20.0,
            self._ask_depth / 20.0,
        ], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    @property
    def portfolio_value(self):
        return self.cash + self.inventory * self.mid


# ── 向量化多环境（用于 PPO rollout）────────────────────────

class VecLOBEnv:
    """
    n_envs 个 LOBEnv 并行运行（CPU，numpy）。
    接口模仿 gymnasium VectorEnv。
    """

    def __init__(self, n_envs: int, env_kwargs: dict):
        self.n_envs = n_envs
        self.envs   = [LOBEnv(**{**env_kwargs, "seed": i * 100 + 1})
                       for i in range(n_envs)]
        self.obs_dim = LOBEnv.OBS_DIM

    def reset(self, seed=None):
        obs_list = []
        for i, env in enumerate(self.envs):
            s = (seed + i) if seed is not None else None
            o, _ = env.reset(seed=s)
            obs_list.append(o)
        return np.stack(obs_list)

    def step(self, actions):
        """actions: (n_envs, 4)"""
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(actions[i])
            done = term or trunc
            if done:
                o, _ = env.reset()
            obs_list.append(o)
            rew_list.append(r)
            done_list.append(done)
            info_list.append(info)
        return (np.stack(obs_list),
                np.array(rew_list, dtype=np.float32),
                np.array(done_list, dtype=bool),
                info_list)
