from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from as_backbone import ASBackbone
from config import CFG, Config
from mbt_factory import make_trading_env
from mbt_gym.gym.index_names import ASSET_PRICE_INDEX, CASH_INDEX, INVENTORY_INDEX, TIME_INDEX


def _to_obs_1d(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 2:
        return arr[0].astype(np.float32)
    return arr.astype(np.float32)


class _BaseMbtAdapter(gym.Env):
    """Gymnasium adapter for local mbt_gym TradingEnvironment (single trajectory)."""

    metadata = {"render_modes": []}

    def __init__(self, base_env, cfg: Optional[Config] = None):
        super().__init__()
        self.base = base_env
        self.cfg = cfg or CFG

        self.action_space = spaces.Box(
            low=np.asarray(self.base.action_space.low, dtype=np.float32),
            high=np.asarray(self.base.action_space.high, dtype=np.float32),
            shape=self.base.action_space.shape,
            dtype=np.float32,
        )
        obs_dim = int(np.asarray(self.base.observation_space.shape).reshape(-1)[0])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._prev_inventory = 0.0
        self._wealth0 = 0.0

    def _state_snapshot(self) -> Tuple[float, float, float, float]:
        s = self.base.state[0]
        cash = float(s[CASH_INDEX])
        inv = float(s[INVENTORY_INDEX])
        t = float(s[TIME_INDEX])
        mid = float(s[ASSET_PRICE_INDEX])
        return cash, inv, t, mid

    def _step_base(self, action_1d: np.ndarray):
        action_2d = np.asarray(action_1d, dtype=np.float32).reshape(1, -1)
        obs, reward, done_arr, _infos = self.base.step(action_2d)
        obs1 = _to_obs_1d(obs)
        reward_scalar = float(np.asarray(reward).reshape(-1)[0])
        terminated = bool(np.asarray(done_arr).reshape(-1)[0])

        cash, inv, t, mid = self._state_snapshot()
        pnl = cash + inv * mid - self._wealth0
        n_fills = float(abs(inv - self._prev_inventory))
        self._prev_inventory = inv

        info = {
            "cash": cash,
            "inventory": inv,
            "time": t,
            "mid_price": mid,
            "pnl": float(pnl),
            "n_fills": n_fills,
        }
        return obs1, reward_scalar, terminated, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None and hasattr(self.base, "seed"):
            self.base.seed(seed)
        obs = self.base.reset()
        cash, inv, _t, mid = self._state_snapshot()
        self._wealth0 = cash + inv * mid
        self._prev_inventory = inv
        return _to_obs_1d(obs), {}

    def step(self, action: np.ndarray):
        return self._step_base(action)

    def close(self):
        self.base.close()


class ModelFreeEnv(_BaseMbtAdapter):
    pass


class HybridTuningEnv(_BaseMbtAdapter):
    """Action = relative adjustments to AS gamma and k."""

    def __init__(self, base_env, cfg: Optional[Config] = None):
        super().__init__(base_env, cfg)
        self._mid = self.cfg.mid
        self._asc = self.cfg.as_cfg
        self._cc = self.cfg.cmdp

        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5], dtype=np.float32),
            high=np.array([0.5, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

        self._backbone = ASBackbone(
            risk_aversion=self._asc.risk_aversion,
            fill_exponent=self.cfg.fill.fill_exponent,
            volatility=self._mid.volatility,
            terminal_time=self._mid.terminal_time,
            tick_size=0.01,
        )

        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_dim + 3,),
            dtype=np.float32,
        )

    def _augment(self, base_obs: np.ndarray) -> np.ndarray:
        _cash, inv, t, _mid = self._state_snapshot()
        tau = max(self._mid.terminal_time - t, 1e-8)
        bid_d, ask_d = self._backbone.depths(inv, tau)
        delta = self._backbone.gueant_spread(tau)
        extra = np.array([bid_d / 10.0, ask_d / 10.0, delta / 10.0], dtype=np.float32)
        return np.concatenate([base_obs.astype(np.float32), extra])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._backbone.gamma = self._asc.risk_aversion
        self._backbone.k = self.cfg.fill.fill_exponent
        return self._augment(obs), info

    def step(self, action: np.ndarray):
        dg, dk = float(action[0]), float(action[1])
        self._backbone.gamma = float(np.clip(self._asc.risk_aversion * (1.0 + dg), 0.01, 2.0))
        self._backbone.k = float(np.clip(self.cfg.fill.fill_exponent * (1.0 + dk), 0.1, 10.0))

        _cash, inv, t, _mid = self._state_snapshot()
        tau = max(self._mid.terminal_time - t, 1e-8)
        bid_d, ask_d = self._backbone.depths(inv, tau)

        obs, base_reward, terminated, truncated, info = self._step_base(np.array([bid_d, ask_d], dtype=np.float32))
        reward = float(base_reward - self._cc.lambda_inventory * inv * inv)
        return self._augment(obs), reward, terminated, truncated, info


class HybridASRLEnv(_BaseMbtAdapter):
    """Residual-action hybrid AS+RL wrapper over mbt_gym TradingEnvironment."""

    def __init__(self, base_env, cfg: Optional[Config] = None, lagrangian_eta: Optional[np.ndarray] = None):
        super().__init__(base_env, cfg)
        self.eta = lagrangian_eta.copy() if lagrangian_eta is not None else np.zeros(2, dtype=np.float64)

        self._rc = self.cfg.residual
        self._cc = self.cfg.cmdp
        self._mid = self.cfg.mid
        self._envc = self.cfg.env
        self._asc = self.cfg.as_cfg
        self._ab = self.cfg.ablation
        self.constraint_names = self._constraint_names()

        rho_bound = self._rc.rho_max if self._ab.use_residual_rho else 0.0
        self.action_space = spaces.Box(
            low=np.array([
                -self._rc.delta_max if self._ab.use_residual_delta else 0.0,
                -self._rc.kappa_max if self._ab.use_residual_kappa else 0.0,
                -rho_bound,
            ], dtype=np.float32),
            high=np.array([
                self._rc.delta_max if self._ab.use_residual_delta else 0.0,
                self._rc.kappa_max if self._ab.use_residual_kappa else 0.0,
                rho_bound,
            ], dtype=np.float32),
            dtype=np.float32,
        )

        self.backbone = ASBackbone(
            risk_aversion=self._asc.risk_aversion,
            fill_exponent=self.cfg.fill.fill_exponent,
            volatility=self._mid.volatility,
            terminal_time=self._mid.terminal_time,
            tick_size=0.01,
        )

        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_dim + 5,),
            dtype=np.float32,
        )

        self._price_window = deque(maxlen=20)
        self._reward_history = []
        self._pnl_history = []
        self._inv_history = []

    def _constraint_names(self):
        mode = str(self._cc.constraint_mode).lower()
        if mode == "risk":
            return ("max_drawdown", "cvar_5pct")
        if mode == "inventory":
            return ("mean_abs_inv", "inv_variance")
        if mode == "hybrid":
            return ("mean_abs_inv", "max_drawdown", "cvar_5pct")
        raise ValueError(f"Unknown CMDP constraint_mode: {self._cc.constraint_mode}")

    def constraint_bounds(self) -> np.ndarray:
        mapping = {
            "max_drawdown": self._cc.drawdown_max,
            "cvar_5pct": self._cc.cvar_epsilon,
            "mean_abs_inv": self._cc.inventory_mean_abs_max,
            "inv_variance": self._cc.inventory_var_max,
        }
        return np.asarray([mapping[name] for name in self.constraint_names], dtype=np.float64)

    def _realised_vol(self) -> float:
        if len(self._price_window) < 2:
            return self._mid.volatility
        prices = np.asarray(self._price_window, dtype=float)
        rets = np.diff(np.log(np.maximum(prices, 1e-8)))
        step_size = self._mid.terminal_time / self._envc.n_steps
        return float(np.std(rets) / np.sqrt(max(step_size, 1e-8)) + 1e-8)

    def _max_drawdown(self) -> float:
        if len(self._pnl_history) < 2:
            return 0.0
        pnl = np.asarray(self._pnl_history, dtype=float)
        peak = np.maximum.accumulate(pnl)
        return float(np.max(peak - pnl))

    def _cvar(self) -> float:
        if len(self._reward_history) < 10:
            return 0.0
        neg = -np.asarray(self._reward_history, dtype=float)
        var = np.quantile(neg, 1.0 - self._cc.cvar_alpha)
        tail = neg[neg >= var]
        return float(np.mean(tail)) if len(tail) else 0.0

    def _inventory_shield(self, bid_d: float, ask_d: float, inventory: float) -> Tuple[float, float]:
        if not self._ab.use_inventory_shield:
            return bid_d, ask_d
        i_max = self._envc.max_inventory
        frac = abs(inventory) / (i_max + 1e-8)
        if frac > 0.8:
            extra = (frac - 0.8) * 0.05
            if inventory > 0:
                ask_d = max(ask_d - extra, 0.01)
                bid_d = bid_d + extra
            else:
                bid_d = max(bid_d - extra, 0.01)
                ask_d = ask_d + extra
        return bid_d, ask_d

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        _cash, inv, t, _mid = self._state_snapshot()
        tau = max(self._mid.terminal_time - t, 1e-8)
        self.backbone.sigma = self._realised_vol() if self._ab.use_adaptive_vol else self._mid.volatility
        bid_d, ask_d = self.backbone.depths(inv, tau)
        delta = self.backbone.gueant_spread(tau)
        eta_pad = np.zeros(2, dtype=np.float32)
        eta_pad[: min(2, len(self.eta))] = self.eta[: min(2, len(self.eta))]
        extra = np.array([bid_d / 10.0, ask_d / 10.0, delta / 10.0, eta_pad[0] / 5.0, eta_pad[1] / 5.0], dtype=np.float32)
        return np.concatenate([base_obs.astype(np.float32), extra])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        base_obs, info = super().reset(seed=seed, options=options)
        self._price_window.clear()
        self._price_window.append(self._mid.initial_price)
        self._reward_history.clear()
        self._pnl_history.clear()
        self._inv_history.clear()
        return self._augment_obs(base_obs), info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        delta_rl, kappa_rl, _rho_rl = float(action[0]), float(action[1]), float(action[2])

        _cash, inv, t, _mid = self._state_snapshot()
        tau = max(self._mid.terminal_time - t, 1e-8)
        sigma_hat = self._realised_vol() if self._ab.use_adaptive_vol else self._mid.volatility
        self.backbone.sigma = sigma_hat

        as_bid_d, as_ask_d = self.backbone.depths(inv, tau)
        as_delta = float(self.backbone.gueant_spread(tau))

        delta_rl = delta_rl if self._ab.use_residual_delta else 0.0
        kappa_rl = kappa_rl if self._ab.use_residual_kappa else 0.0

        delta_t = max(as_delta + delta_rl, 0.01)
        gamma_eff = max(self._asc.risk_aversion + kappa_rl, 1e-6)
        skew = inv * gamma_eff * sigma_hat * sigma_hat * tau
        bid_depth = max(delta_t / 2.0 + skew, 0.01)
        ask_depth = max(delta_t / 2.0 - skew, 0.01)
        bid_depth, ask_depth = self._inventory_shield(bid_depth, ask_depth, inv)

        base_obs, base_reward, terminated, truncated, info = self._step_base(np.array([bid_depth, ask_depth], dtype=np.float32))

        mid_price = float(info.get("mid_price", self._mid.initial_price))
        self._price_window.append(mid_price)

        inv_penalty = self._cc.lambda_inventory * inv * inv
        adv_sel = 0.0
        if len(self._price_window) >= 2:
            adv_sel = abs(self._price_window[-1] - self._price_window[-2]) * info.get("n_fills", 0.0)
        fee_cost = self._cc.fee_per_trade * info.get("n_fills", 0.0)

        raw_reward = float(base_reward - inv_penalty - self._cc.lambda_adverse * adv_sel - fee_cost)
        self._reward_history.append(raw_reward)
        self._pnl_history.append(float(info.get("pnl", 0.0)))
        self._inv_history.append(inv)

        g = self._constraint_vector()
        shaped_reward = float(raw_reward - float(np.dot(self.eta, g))) if self._ab.use_cmdp else raw_reward

        info.update(
            {
                "as_bid_depth": float(as_bid_d),
                "as_ask_depth": float(as_ask_d),
                "as_spread": as_delta,
                "hybrid_bid_depth": float(bid_depth),
                "hybrid_ask_depth": float(ask_depth),
                "raw_reward": raw_reward,
                "constraint_g": g,
                "constraint_names": self.constraint_names,
            }
        )
        return self._augment_obs(base_obs), shaped_reward, terminated, truncated, info

    def set_lagrangian_eta(self, eta: np.ndarray) -> None:
        self.eta = np.asarray(eta, dtype=np.float64).copy()

    def episode_summary(self) -> Dict[str, float]:
        rewards = np.asarray(self._reward_history, dtype=float) if self._reward_history else np.zeros(1)
        summary = {
            "episode_raw_reward": float(np.sum(rewards)),
            "max_drawdown": self._max_drawdown(),
            "cvar_5pct": self._cvar(),
            "mean_abs_inv": float(np.mean(np.abs(self._inv_history))) if self._inv_history else 0.0,
            "inv_variance": float(np.var(self._inv_history)) if self._inv_history else 0.0,
        }
        for idx, name in enumerate(self.constraint_names):
            summary[f"constraint_{idx}"] = float(summary[name])
        return summary

    def _constraint_vector(self) -> np.ndarray:
        summary = self.episode_summary()
        return np.asarray([summary[name] for name in self.constraint_names], dtype=np.float64)


def make_model_free_env(cfg: Optional[Config] = None, seed: int = 0, regime: str = "standard") -> ModelFreeEnv:
    return ModelFreeEnv(make_trading_env(cfg=cfg, seed=seed, regime=regime), cfg=cfg)


def make_hybrid_tuning_env(cfg: Optional[Config] = None, seed: int = 0, regime: str = "standard") -> HybridTuningEnv:
    return HybridTuningEnv(make_trading_env(cfg=cfg, seed=seed, regime=regime), cfg=cfg)


def make_hybrid_env(
    cfg: Optional[Config] = None,
    seed: int = 0,
    regime: str = "standard",
    eta: Optional[np.ndarray] = None,
) -> HybridASRLEnv:
    return HybridASRLEnv(make_trading_env(cfg=cfg, seed=seed, regime=regime), cfg=cfg, lagrangian_eta=eta)
