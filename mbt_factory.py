"""
Factory for building mbt_gym TradingEnvironment instances.
Compatible with the local mbt_gym source layout in this workspace.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import CFG, Config

try:
    from mbt_gym.gym.ModelDynamics import LimitOrderModelDynamics
    from mbt_gym.gym.TradingEnvironment import TradingEnvironment
    from mbt_gym.rewards.RewardFunctions import PnL
    from mbt_gym.stochastic_processes.arrival_models import PoissonArrivalModel
    from mbt_gym.stochastic_processes.fill_probability_models import ExponentialFillFunction
    from mbt_gym.stochastic_processes.midprice_models import BrownianMotionMidpriceModel

    _MBT_AVAILABLE = True
except ImportError:
    _MBT_AVAILABLE = False
    TradingEnvironment = None


def make_trading_env(
    cfg: Optional[Config] = None,
    seed: Optional[int] = None,
    regime: str = "standard",  # standard | volatile | thin
):
    cfg = cfg or CFG
    if not _MBT_AVAILABLE:
        raise ImportError("mbt_gym import failed. Check local mbt_gym package path.")

    if regime not in {"standard", "volatile", "thin"}:
        raise ValueError(f"Unknown regime: {regime}")

    vol_scale = {"standard": 1.0, "volatile": 2.0, "thin": 1.0}[regime]
    arr_scale = {"standard": 1.0, "volatile": 1.0, "thin": 0.5}[regime]

    step_size = cfg.mid.terminal_time / cfg.env.n_steps

    midprice_model = BrownianMotionMidpriceModel(
        volatility=cfg.mid.volatility * vol_scale,
        step_size=step_size,
        terminal_time=cfg.mid.terminal_time,
        initial_price=cfg.mid.initial_price,
        num_trajectories=cfg.env.num_trajectories,
        seed=seed,
    )

    arrival_model = PoissonArrivalModel(
        intensity=np.array(cfg.arrival.intensity, dtype=float) * arr_scale,
        step_size=step_size,
        num_trajectories=cfg.env.num_trajectories,
        seed=seed,
    )

    fill_probability_model = ExponentialFillFunction(
        fill_exponent=cfg.fill.fill_exponent,
        step_size=step_size,
        num_trajectories=cfg.env.num_trajectories,
        seed=seed,
    )

    model_dynamics = LimitOrderModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        fill_probability_model=fill_probability_model,
        num_trajectories=cfg.env.num_trajectories,
        seed=seed,
    )

    env = TradingEnvironment(
        terminal_time=cfg.mid.terminal_time,
        n_steps=cfg.env.n_steps,
        reward_function=PnL(),
        model_dynamics=model_dynamics,
        initial_inventory=cfg.env.initial_inventory,
        max_inventory=cfg.env.max_inventory,
        num_trajectories=cfg.env.num_trajectories,
        seed=seed,
        normalise_action_space=False,
        normalise_observation_space=False,
    )
    return env


def get_obs_dim(cfg: Optional[Config] = None) -> int:
    cfg = cfg or CFG
    env = make_trading_env(cfg, seed=0)
    obs = env.reset()
    dim = int(obs.shape[-1])
    env.close()
    return dim
