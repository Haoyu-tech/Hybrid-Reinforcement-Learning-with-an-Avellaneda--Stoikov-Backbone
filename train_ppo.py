from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

# Make torch/sb3 import robust on this Windows setup (avoid CUDA DLL init issues).
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm.auto import tqdm

from config import CFG, CMDPConfig, Config
from hybrid_env import HybridASRLEnv, make_hybrid_env, make_hybrid_tuning_env, make_model_free_env


class LagrangianCMDPCallback(BaseCallback):
    def __init__(self, envs: List[HybridASRLEnv], cmdp_cfg: Optional[CMDPConfig] = None, verbose: int = 0):
        super().__init__(verbose)
        self.envs = envs
        self.cc = cmdp_cfg or CFG.cmdp
        self.constraint_names = tuple(envs[0].constraint_names) if envs else ()
        self.eta = np.full(len(self.constraint_names), self.cc.eta_init, dtype=np.float64)
        self.d = envs[0].constraint_bounds() if envs else np.zeros(0, dtype=np.float64)
        self.eta_history: List[np.ndarray] = []

    def _on_rollout_end(self) -> None:
        g_list = []
        for env in self.envs:
            s = env.episode_summary()
            g_list.append([s[name] for name in self.constraint_names])
        if not g_list:
            return

        g_bar = np.mean(g_list, axis=0)
        self.eta = np.maximum(0.0, self.eta + self.cc.eta_lr * (g_bar - self.d))
        self.eta_history.append(self.eta.copy())

        for env in self.envs:
            env.set_lagrangian_eta(self.eta)

        if self.verbose >= 1:
            print(f"[CMDP] names={self.constraint_names} eta={self.eta.round(4)} g={np.asarray(g_bar).round(4)}")

    def _on_step(self) -> bool:
        return True


class TqdmProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, desc: str, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.desc = desc
        self.pbar = None
        self.last_n = 0

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc=self.desc, dynamic_ncols=True)
        self.last_n = 0

    def _on_step(self) -> bool:
        if self.pbar is None:
            return True
        current = min(int(self.model.num_timesteps), self.total_timesteps)
        delta = current - self.last_n
        if delta > 0:
            self.pbar.update(delta)
            self.last_n = current
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            if self.last_n < self.total_timesteps:
                self.pbar.update(self.total_timesteps - self.last_n)
            self.pbar.close()
            self.pbar = None


def _build_single_env(env_type: str, cfg: Config, seed: int, raw_hybrid_envs: Optional[List[HybridASRLEnv]] = None):
    set_random_seed(seed)
    if env_type == "hybrid":
        raw = make_hybrid_env(cfg=cfg, seed=seed)
        if raw_hybrid_envs is not None:
            raw_hybrid_envs.append(raw)
        return Monitor(raw)
    if env_type == "model_free":
        return Monitor(make_model_free_env(cfg=cfg, seed=seed))
    if env_type == "hybrid_tuning":
        return Monitor(make_hybrid_tuning_env(cfg=cfg, seed=seed))
    raise ValueError(f"Unknown env_type: {env_type}")


def _make_unique_log_tag(tag: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{tag}_{timestamp}_{os.getpid()}"


def _save_model_with_retry(model: PPO, save_path: str, verbose: int = 1) -> str:
    candidates = [save_path]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidates.append(f"{save_path}_{timestamp}_{os.getpid()}")

    last_err: Optional[Exception] = None
    for idx, candidate in enumerate(candidates):
        try:
            model.save(candidate)
            if verbose >= 1:
                label = "Saved model" if idx == 0 else "Saved model to fallback path"
                print(f"{label}: {candidate}.zip")
            return candidate
        except PermissionError as e:
            last_err = e
            if verbose >= 1:
                print(f"[WARN] Save blocked for {candidate}.zip: {e}")
            time.sleep(1.0)

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to save model to {save_path}")


def train_ppo(
    env_type: str = "hybrid",
    cfg: Optional[Config] = None,
    seed: Optional[int] = None,
    run_name: str = "",
    use_lagrangian: bool = True,
    verbose: int = 1,
    regime: str = "standard",
):
    cfg = cfg or CFG
    tc = cfg.train
    seed = tc.seed if seed is None else seed

    os.makedirs(tc.log_dir, exist_ok=True)
    os.makedirs(tc.model_dir, exist_ok=True)

    tag = f"{env_type}_{regime}_seed{seed}" + (f"_{run_name}" if run_name else "")
    log_tag = _make_unique_log_tag(tag)
    run_log_dir = os.path.join(tc.log_dir, log_tag)
    checkpoint_dir = os.path.join(tc.model_dir, "checkpoints", log_tag)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    raw_hybrid_envs: List[HybridASRLEnv] = []

    train_vec = DummyVecEnv(
        [
            (lambda i=i: Monitor(
                make_hybrid_env(cfg=cfg, seed=seed + i, regime=regime)
            ) if env_type == "hybrid" else
            Monitor(make_model_free_env(cfg=cfg, seed=seed + i, regime=regime)) if env_type == "model_free" else
            Monitor(make_hybrid_tuning_env(cfg=cfg, seed=seed + i, regime=regime)))
            for i in range(tc.n_envs)
        ]
    )
    if env_type == "hybrid":
        for i in range(tc.n_envs):
            raw_hybrid_envs.append(train_vec.envs[i].env)
        eval_vec = DummyVecEnv([lambda: Monitor(make_hybrid_env(cfg=cfg, seed=seed + tc.n_envs, regime=regime))])
    elif env_type == "model_free":
        eval_vec = DummyVecEnv([lambda: Monitor(make_model_free_env(cfg=cfg, seed=seed + tc.n_envs, regime=regime))])
    else:
        eval_vec = DummyVecEnv([lambda: Monitor(make_hybrid_tuning_env(cfg=cfg, seed=seed + tc.n_envs, regime=regime))])

    model = PPO(
        policy="MlpPolicy",
        env=train_vec,
        learning_rate=tc.learning_rate,
        n_steps=tc.n_steps,
        batch_size=tc.batch_size,
        n_epochs=tc.n_epochs,
        gamma=tc.gamma_discount,
        gae_lambda=tc.gae_lambda,
        clip_range=tc.clip_range,
        ent_coef=tc.ent_coef,
        vf_coef=tc.vf_coef,
        max_grad_norm=tc.max_grad_norm,
        policy_kwargs={"net_arch": tc.net_arch, "activation_fn": nn.Tanh},
        tensorboard_log=None,
        verbose=verbose,
        seed=seed,
        device="auto",
    )

    callbacks = [
        TqdmProgressCallback(
            total_timesteps=tc.total_timesteps,
            desc=f"{env_type} seed={seed}",
        ),
        EvalCallback(
            eval_env=eval_vec,
            best_model_save_path=None,
            log_path=None,
            eval_freq=max(tc.n_steps * tc.n_envs, 10_000),
            n_eval_episodes=tc.n_eval_episodes,
            deterministic=True,
            verbose=0,
        ),
        CheckpointCallback(
            save_freq=max(100_000 // max(tc.n_envs, 1), tc.n_steps),
            save_path=checkpoint_dir,
            name_prefix=tag,
            verbose=0,
        ),
    ]

    if use_lagrangian and cfg.ablation.use_cmdp and env_type == "hybrid" and raw_hybrid_envs:
        callbacks.append(LagrangianCMDPCallback(envs=raw_hybrid_envs, cmdp_cfg=cfg.cmdp, verbose=max(verbose - 1, 0)))

    if verbose >= 1:
        print(f"\n{'=' * 60}")
        print(f"PPO training | env={env_type} | regime={regime} | seed={seed}")
        print(f"Timesteps: {tc.total_timesteps:,} | n_envs: {tc.n_envs}")
        print(f"{'=' * 60}")

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=tc.total_timesteps,
            callback=callbacks,
            tb_log_name=None,
            reset_num_timesteps=True,
            progress_bar=False,
        )

        if verbose >= 1:
            print(f"Done in {time.time() - t0:.1f}s")
            print(f"Checkpoint dir: {checkpoint_dir}")

        save_path = os.path.join(tc.model_dir, f"ppo_{tag}_final")
        _save_model_with_retry(model, save_path, verbose=verbose)
    finally:
        if hasattr(model, "logger") and model.logger is not None:
            model.logger.close()
        train_vec.close()
        eval_vec.close()

    return model


def train_all(cfg: Optional[Config] = None, n_seeds: int = 5) -> Dict[str, List[PPO]]:
    cfg = cfg or CFG
    out: Dict[str, List[PPO]] = {}

    for env_type in ["hybrid", "model_free", "hybrid_tuning"]:
        models: List[PPO] = []
        for i in range(n_seeds):
            seed = cfg.train.seed + i * 100
            regime = cfg.eval.regimes[0]
            print(f"\n[{env_type} {regime} seed {i + 1}/{n_seeds}]")
            m = train_ppo(
                env_type=env_type,
                cfg=cfg,
                seed=seed,
                use_lagrangian=(env_type == "hybrid"),
                verbose=1,
                regime=regime,
            )
            models.append(m)
        out[env_type] = models

    return out
