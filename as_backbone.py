"""
environments/as_backbone.py
============================
Avellaneda–Stoikov backbone.

mbt-gym ships its own AvellanedaStoikovAgent in
    mbt_gym.agents.BaselineAgents.AvellanedaStoikovAgent

We re-implement the same formulae here so that:
  (a) the backbone can run inside a gym.Wrapper without needing
      a reference to the outer TradingEnvironment, and
  (b) we can output *residual-ready* (bid_depth, ask_depth) directly.

The math is identical to AvellanedaStoikovAgent — no invention.

Notation follows the paper (Section 3 / 5.2):
    r_t   = S_t - I_t · γ · σ̂² · τ        (reservation price)
    Δ_AS  = γ · σ̂² · τ + (2/γ) · ln(1 + γ/k)   (Guéant spread)
    δ^b   = Δ_AS/2 + I_t · γ · σ̂² · τ     (bid depth from mid)
    δ^a   = Δ_AS/2 - I_t · γ · σ̂² · τ     (ask depth from mid)
"""

import numpy as np
from typing import Union

ArrayLike = Union[float, np.ndarray]


class ASBackbone:
    """
    Pure-math AS backbone.  Mirrors the logic in
    mbt_gym.agents.BaselineAgents.AvellanedaStoikovAgent but is
    self-contained so it can run inside any gym.Wrapper.

    Parameters
    ----------
    risk_aversion : float
        γ (CARA coefficient).
    fill_exponent : float
        k in P(fill|δ) = exp(−k·δ).  Shared with ExponentialFillFunction.
    volatility : float
        σ (per-step or annualised, must match midprice_model.volatility).
    terminal_time : float
        T (normalised to 1 in mbt-gym).
    tick_size : float
        Minimum price increment; quotes rounded to this.
    """

    def __init__(
        self,
        risk_aversion:  float = 0.1,
        fill_exponent:  float = 1.5,
        volatility:     float = 2.0,
        terminal_time:  float = 1.0,
        tick_size:      float = 0.01,
    ):
        self.gamma        = risk_aversion
        self.k            = fill_exponent
        self.sigma        = volatility
        self.T            = terminal_time
        self.tick_size    = tick_size

    # ── core formulas ────────────────────────────────────────────────────

    def reservation_price(
        self,
        mid_price: ArrayLike,
        inventory: ArrayLike,
        time_remaining: ArrayLike,
    ) -> ArrayLike:
        """r_t = S_t − I_t · γ · σ² · τ   (paper Eq. 9)"""
        tau = np.clip(time_remaining, 1e-8, self.T)
        return mid_price - inventory * self.gamma * self.sigma**2 * tau

    def gueant_spread(self, time_remaining: ArrayLike) -> ArrayLike:
        """
        Guéant et al. (2013) spread approximation:
            Δ_AS = γ σ² τ + (2/γ) ln(1 + γ/k)
        """
        tau = np.clip(time_remaining, 1e-8, self.T)
        return (self.gamma * self.sigma**2 * tau
                + (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.k))

    def depths(
        self,
        inventory: ArrayLike,
        time_remaining: ArrayLike,
    ) -> tuple:
        """
        AS optimal bid and ask depths from mid-price.

        δ^b = Δ_AS/2 + I · γ · σ² · τ
        δ^a = Δ_AS/2 − I · γ · σ² · τ

        Both clipped to [tick_size, ∞).

        Returns
        -------
        (bid_depth, ask_depth) : tuple of ArrayLike
            Positive depths.  Pass directly to TradingEnvironment.step().
        """
        tau   = np.clip(time_remaining, 1e-8, self.T)
        delta = self.gueant_spread(tau)
        skew  = inventory * self.gamma * self.sigma**2 * tau

        bid_depth = np.maximum(delta / 2.0 + skew, self.tick_size)
        ask_depth = np.maximum(delta / 2.0 - skew, self.tick_size)
        return bid_depth, ask_depth

    # ── observation extraction helpers ──────────────────────────────────

    @staticmethod
    def extract_from_obs(
        obs: np.ndarray,
        max_inventory: int,
        terminal_time: float,
        initial_price: float,
    ) -> tuple:
        """
        Extract (time_remaining, inventory, mid_price) from a
        TradingEnvironment observation vector.

        mbt-gym default obs layout (BrownianMotion + Poisson, no FeatureEngineer):
            obs[0] = time elapsed / terminal_time          (∈ [0,1])
            obs[1] = inventory / max_inventory             (∈ [-1,1])
            obs[2] = mid_price / initial_price             (normalised)

        Returns
        -------
        (time_remaining, inventory_raw, mid_price_raw)
        """
        time_elapsed   = float(obs[0]) * terminal_time
        time_remaining = max(terminal_time - time_elapsed, 1e-8)
        inventory      = float(obs[1]) * max_inventory
        mid_price      = float(obs[2]) * initial_price
        return time_remaining, inventory, mid_price

    def action_from_obs(
        self,
        obs: np.ndarray,
        max_inventory: int,
        initial_price: float,
    ) -> np.ndarray:
        """
        Convenience: compute AS depths directly from a TradingEnvironment obs.
        Returns np.array([bid_depth, ask_depth]).
        """
        tau, inv, _ = self.extract_from_obs(
            obs, max_inventory, self.T, initial_price
        )
        bid_d, ask_d = self.depths(inv, tau)
        return np.array([bid_d, ask_d], dtype=np.float32)
