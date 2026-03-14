"""
config.py
=========
Central configuration.  Parameter names intentionally mirror mbt-gym's
own constructor signatures so they can be passed in directly.

mbt-gym reference: Jerome et al. (2023), ICAIF
GitHub: https://github.com/JeromeJeromeTomczak/mbt-gym
"""

from dataclasses import dataclass, field
from typing import Tuple


# ── mbt-gym BrownianMotionMidpriceModel params ─────────────────────────────
@dataclass
class MidpriceConfig:
    volatility: float     = 2.0       # σ  — annual vol in price-tick units
    dt: float             = 1 / 200   # time step (fraction of trading day)
    terminal_time: float  = 1.0       # T  (1 = full trading day)
    initial_price: float  = 100.0


# ── mbt-gym PoissonArrivalModel params ────────────────────────────────────
@dataclass
class ArrivalConfig:
    # shape (2,): [bid_intensity, ask_intensity]  A in λ = A·exp(-k·δ)
    intensity: Tuple[float, float] = (140.0, 140.0)


# ── mbt-gym ExponentialFillFunction params ────────────────────────────────
@dataclass
class FillConfig:
    fill_exponent: float = 1.5        # k in P(fill|δ) = exp(-k·δ)


# ── mbt-gym TradingEnvironment params ─────────────────────────────────────
@dataclass
class EnvConfig:
    max_inventory: int    = 10
    n_steps: int          = 200       # steps per episode  (= T/dt = 200)
    initial_inventory: int= 0
    num_trajectories: int = 1         # vectorised envs handled by SB3 VecEnv


# ── Avellaneda–Stoikov backbone ────────────────────────────────────────────
@dataclass
class ASConfig:
    risk_aversion: float  = 0.1       # γ
    # k (depth sensitivity) is shared with FillConfig.fill_exponent
    # H (horizon) is computed from terminal_time dynamically


# ── RL residual bounds (paper Eqs. 11–14) ─────────────────────────────────
@dataclass
class ResidualConfig:
    delta_max: float = 0.05    # max spread adjustment  (price units)
    kappa_max: float = 0.05    # max skew adjustment
    rho_max:   float = 0.50    # max log-size scaling
    n_levels:  int   = 1       # how many limit-order levels per side


@dataclass
class AblationConfig:
    use_cmdp: bool = True
    use_inventory_shield: bool = True
    use_adaptive_vol: bool = True
    use_residual_delta: bool = True
    use_residual_kappa: bool = True
    use_residual_rho: bool = False


# ── CMDP / reward penalties ────────────────────────────────────────────────
@dataclass
class CMDPConfig:
    lambda_inventory:  float = 0.1    # quadratic inventory penalty
    lambda_adverse:    float = 0.05   # adverse-selection penalty
    drawdown_max:      float = 5.0    # E[MaxDD] constraint bound
    cvar_epsilon:      float = 10.0   # CVaR_α constraint bound
    inventory_mean_abs_max: float = 2.0
    inventory_var_max: float = 9.0
    cvar_alpha:        float = 0.05
    fee_per_trade:     float = 0.0002 # maker fee/rebate (fraction of mid)
    cancel_cost:       float = 0.0001
    constraint_mode:   str   = "inventory"
    # Lagrangian dual ascent
    eta_lr:            float = 0.01
    eta_init:          float = 0.0


# ── SB3 PPO training ──────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    total_timesteps:  int   = 500_000
    n_envs:           int   = 4
    n_steps:          int   = 2048
    batch_size:       int   = 256
    n_epochs:         int   = 10
    learning_rate:    float = 3e-4
    gamma_discount:   float = 0.99
    gae_lambda:       float = 0.95
    clip_range:       float = 0.2
    ent_coef:         float = 0.01
    vf_coef:          float = 0.5
    max_grad_norm:    float = 0.5
    net_arch:         list  = field(default_factory=lambda: [256, 256])
    log_dir:          str   = "outputs/logs"
    model_dir:        str   = "outputs/models"
    seed:             int   = 42
    n_eval_episodes:  int   = 20


# ── Evaluation ────────────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    n_seeds:        int = 5
    eval_episodes:  int = 50
    base_seed:      int = 999
    train_seed_stride: int = 100
    eval_seed_stride: int = 13
    regimes: Tuple[str, ...] = ("standard", "volatile", "thin")
    results_dir:    str = "outputs/results"
    figures_dir:    str = "outputs/figures"


# ── Master config ─────────────────────────────────────────────────────────
@dataclass
class Config:
    mid:      MidpriceConfig  = field(default_factory=MidpriceConfig)
    arrival:  ArrivalConfig   = field(default_factory=ArrivalConfig)
    fill:     FillConfig      = field(default_factory=FillConfig)
    env:      EnvConfig       = field(default_factory=EnvConfig)
    as_cfg:   ASConfig        = field(default_factory=ASConfig)
    residual: ResidualConfig  = field(default_factory=ResidualConfig)
    ablation: AblationConfig  = field(default_factory=AblationConfig)
    cmdp:     CMDPConfig      = field(default_factory=CMDPConfig)
    train:    TrainConfig     = field(default_factory=TrainConfig)
    eval:     EvalConfig      = field(default_factory=EvalConfig)


CFG = Config()
