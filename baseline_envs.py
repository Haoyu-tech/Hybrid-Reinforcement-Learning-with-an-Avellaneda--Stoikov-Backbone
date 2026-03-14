"""Baseline adapters compatible with local mbt_gym API."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from as_backbone import ASBackbone
from config import CFG, Config
from hybrid_env import HybridTuningEnv, make_hybrid_tuning_env, make_model_free_env
from mbt_factory import make_trading_env
from mbt_gym.agents.BaselineAgents import AvellanedaStoikovAgent


class StaticASAgent:
    """Thin wrapper around mbt_gym AvellanedaStoikovAgent."""

    def __init__(self, env, risk_aversion: Optional[float] = None, cfg: Optional[Config] = None):
        self.cfg = cfg or CFG
        gamma = self.cfg.as_cfg.risk_aversion if risk_aversion is None else risk_aversion
        self.env = env
        self.agent = AvellanedaStoikovAgent(env=env, risk_aversion=gamma)

    def get_action(self, _obs: np.ndarray) -> np.ndarray:
        # mbt_gym baseline agents read env state matrix.
        return self.agent.get_action(self.env.state)

    def run_episode(self, seed: int = 0) -> Dict:
        _obs = self.env.reset()
        done = False
        records = []

        wealth0 = float(self.env.state[0, 0] + self.env.state[0, 1] * self.env.state[0, 3])
        prev_inv = float(self.env.state[0, 1])

        while not done:
            action = self.get_action(_obs)
            _obs, reward, dones, _infos = self.env.step(action)
            done = bool(np.asarray(dones).reshape(-1)[0])

            s = self.env.state[0]
            cash, inv, mid = float(s[0]), float(s[1]), float(s[3])
            records.append(
                {
                    "reward": float(np.asarray(reward).reshape(-1)[0]),
                    "inventory": inv,
                    "pnl": cash + inv * mid - wealth0,
                    "n_fills": abs(inv - prev_inv),
                }
            )
            prev_inv = inv
        return records


class ConstrainedASAgent:
    """AS backbone with clipped effective inventory."""

    def __init__(self, env, inv_cap_frac: float = 0.8, cfg: Optional[Config] = None):
        self.cfg = cfg or CFG
        self.env = env
        self.inv_cap = self.cfg.env.max_inventory * inv_cap_frac
        self.backbone = ASBackbone(
            risk_aversion=self.cfg.as_cfg.risk_aversion,
            fill_exponent=self.cfg.fill.fill_exponent,
            volatility=self.cfg.mid.volatility,
            terminal_time=self.cfg.mid.terminal_time,
            tick_size=0.01,
        )

    def get_action(self, _obs: np.ndarray) -> np.ndarray:
        s = self.env.state[0]
        inv = float(np.clip(s[1], -self.inv_cap, self.inv_cap))
        t = float(s[2])
        tau = max(self.cfg.mid.terminal_time - t, 1e-8)
        bid_d, ask_d = self.backbone.depths(inv, tau)
        return np.array([[bid_d, ask_d]], dtype=np.float32)

    def run_episode(self, seed: int = 0) -> Dict:
        _obs = self.env.reset()
        done = False
        records = []

        wealth0 = float(self.env.state[0, 0] + self.env.state[0, 1] * self.env.state[0, 3])
        prev_inv = float(self.env.state[0, 1])

        while not done:
            action = self.get_action(_obs)
            _obs, reward, dones, _infos = self.env.step(action)
            done = bool(np.asarray(dones).reshape(-1)[0])

            s = self.env.state[0]
            cash, inv, mid = float(s[0]), float(s[1]), float(s[3])
            records.append(
                {
                    "reward": float(np.asarray(reward).reshape(-1)[0]),
                    "inventory": inv,
                    "pnl": cash + inv * mid - wealth0,
                    "n_fills": abs(inv - prev_inv),
                }
            )
            prev_inv = inv
        return records


__all__ = [
    "StaticASAgent",
    "ConstrainedASAgent",
    "HybridTuningEnv",
    "make_model_free_env",
    "make_hybrid_tuning_env",
    "make_trading_env",
]
