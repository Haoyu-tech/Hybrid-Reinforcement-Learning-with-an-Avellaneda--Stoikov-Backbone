"""
PPO agent for minute-bar market-making experiments.
Supports both pure RL and AS+RL hybrid action decoding.
"""

import os
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


@dataclass
class PPOConfig:
    obs_dim: int = 12
    action_dim: int = 4
    policy_mode: str = "hybrid"  # "hybrid" | "model_free"
    hidden_dim: int = 256
    n_hidden: int = 2
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_envs: int = 8
    rollout_steps: int = 512
    n_epochs: int = 10
    batch_size: int = 256
    total_timesteps: int = 100_000
    use_lagrangian: bool = True
    constraint_budget: float = 0.05
    eta_lr: float = 1e-3
    eta_max: float = 10.0
    inventory_limit: float = 10.0
    target_kl: float = 0.02  # KL 超过此值提前结束 epoch，防止策略崩溃
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()

        layers = []
        width = obs_dim
        for _ in range(n_hidden):
            layers.extend([nn.Linear(width, hidden_dim), nn.Tanh()])
            width = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.full((action_dim,), -1.0))  # 初始 std≈0.37，避免大方差导致KL爆炸
        self.critic_head = nn.Linear(hidden_dim, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, obs):
        feat = self.encoder(obs)
        mean = torch.tanh(self.actor_mean(feat))
        logstd = self.actor_logstd.clamp(-3, -0.5)  # std ∈ [0.05, 0.61]，稳定分布
        value = self.critic_head(feat)
        return mean, logstd, value

    def get_action_and_value(self, obs, action=None):
        mean, logstd, value = self.forward(obs)
        dist = Normal(mean, logstd.exp())
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logp, entropy, value.squeeze(-1)


class RolloutBuffer:
    def __init__(self, rollout_steps, n_envs, obs_dim, action_dim, gamma, gae_lambda, device):
        t, n = rollout_steps, n_envs
        self.obs = torch.zeros(t, n, obs_dim)
        self.actions = torch.zeros(t, n, action_dim)
        self.rewards = torch.zeros(t, n)
        self.costs = torch.zeros(t, n)
        self.values = torch.zeros(t, n)
        self.logprobs = torch.zeros(t, n)
        self.dones = torch.zeros(t, n)
        self.advantages = torch.zeros(t, n)
        self.returns = torch.zeros(t, n)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.ptr = 0

    def add(self, obs, actions, rewards, costs, values, logprobs, dones):
        idx = self.ptr
        self.obs[idx] = torch.as_tensor(obs, dtype=torch.float32)
        self.actions[idx] = torch.as_tensor(actions, dtype=torch.float32)
        self.rewards[idx] = torch.as_tensor(rewards, dtype=torch.float32)
        self.costs[idx] = torch.as_tensor(costs, dtype=torch.float32)
        self.values[idx] = torch.as_tensor(values, dtype=torch.float32)
        self.logprobs[idx] = torch.as_tensor(logprobs, dtype=torch.float32)
        self.dones[idx] = torch.as_tensor(dones, dtype=torch.float32)
        self.ptr += 1

    def compute_gae(self, last_value, last_done):
        last_gae = torch.zeros_like(last_value)
        t = self.ptr
        for idx in reversed(range(t)):
            if idx == t - 1:
                next_non_terminal = 1.0 - last_done.float()
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[idx]  # done[idx]=1 means step idx ended episode → no bootstrap
                next_value = self.values[idx + 1]
            delta = self.rewards[idx] + self.gamma * next_value * next_non_terminal - self.values[idx]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[idx] = last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size, eta=0.0):
        t, n = self.ptr, self.obs.shape[1]
        total = t * n
        obs_f = self.obs[:t].reshape(total, -1).to(self.device)
        act_f = self.actions[:t].reshape(total, -1).to(self.device)
        lp_f = self.logprobs[:t].reshape(total).to(self.device)
        adv_f = self.advantages[:t].reshape(total).to(self.device)
        ret_f = self.returns[:t].reshape(total).to(self.device)
        cost_f = self.costs[:t].reshape(total).to(self.device)
        if eta > 0:
            adv_f = adv_f - eta * cost_f
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        perm = torch.randperm(total)
        for start in range(0, total, batch_size):
            idx = perm[start : start + batch_size]
            yield obs_f[idx], act_f[idx], lp_f[idx], adv_f[idx], ret_f[idx], cost_f[idx]

    def reset(self):
        self.ptr = 0


class PPOTrainer:
    def __init__(self, cfg: PPOConfig, backbone=None):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.backbone = backbone

        self.ac = ActorCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim, cfg.n_hidden).to(self.dev)
        self.opt = optim.Adam(self.ac.parameters(), lr=cfg.lr_actor, eps=1e-5)
        total_updates = max(cfg.total_timesteps // (cfg.rollout_steps * cfg.n_envs), 1)
        self.sched = optim.lr_scheduler.LinearLR(
            self.opt,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_updates,
        )

        self.eta = 0.0
        self.global_step = 0
        self.buffer = RolloutBuffer(
            cfg.rollout_steps,
            cfg.n_envs,
            cfg.obs_dim,
            cfg.action_dim,
            cfg.gamma,
            cfg.gae_lambda,
            cfg.device,
        )

    @torch.no_grad()
    def act(self, obs, mid, inventory, t_remaining, sigma_hat, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.dev)
        mean, logstd, value = self.ac(obs_t)
        dist = Normal(mean, logstd.exp())
        raw_action = mean if deterministic else dist.sample()
        logp = dist.log_prob(raw_action).sum(-1)
        # decoded_actions 送给环境执行，raw_action 存入 buffer 供 PPO update 计算 log_prob
        decoded_actions = self._decode_action(raw_action.cpu().numpy(), mid, inventory, t_remaining, sigma_hat)
        return decoded_actions, raw_action.cpu().numpy(), logp.cpu().numpy(), value.squeeze(-1).cpu().numpy()

    def _decode_action(self, raw_action, mid, inventory, t_remaining, sigma_hat):
        if self.cfg.policy_mode == "hybrid":
            return self._hybrid_action(raw_action, mid, inventory, t_remaining, sigma_hat)
        if self.cfg.policy_mode == "model_free":
            return self._model_free_action(raw_action, inventory)
        raise ValueError(f"Unknown policy_mode: {self.cfg.policy_mode}")

    def _model_free_action(self, raw_action, inventory):
        raw_action = np.atleast_2d(raw_action)
        inv_arr = np.atleast_1d(inventory)
        n = raw_action.shape[0]
        tick = 0.01
        min_off = tick
        max_off = 8.0 * tick   # BTC 1m: bar range 中位数0.095，max half-spread=0.08使fill rate~24%
        log_size_lo = np.log(0.1)
        log_size_hi = np.log(2.0)

        actions = np.zeros((n, self.cfg.action_dim), dtype=np.float32)
        spread_scale = (raw_action[:, 0] + 1.0) * 0.5
        half_spread = min_off + spread_scale * (max_off - min_off)
        skew = raw_action[:, 1] * 0.60 * half_spread
        bid_off = np.clip(half_spread + skew, min_off, max_off)
        ask_off = np.clip(half_spread - skew, min_off, max_off)

        bid_off, ask_off, bid_sz_scale, ask_sz_scale = self._inventory_risk_overlay(
            bid_off,
            ask_off,
            np.ones(n, dtype=np.float32),
            np.ones(n, dtype=np.float32),
            inv_arr,
        )
        actions[:, 0] = np.log(np.clip(bid_off, min_off, max_off))
        actions[:, 1] = np.log(np.clip(ask_off, min_off, max_off))

        if self.cfg.action_dim > 2:
            size_scale = (raw_action[:, 2:4] + 1.0) * 0.5
            bid_scale = np.clip(size_scale[:, 0] * bid_sz_scale, 0.0, 1.0)
            ask_scale = np.clip(size_scale[:, 1] * ask_sz_scale, 0.0, 1.0)
            actions[:, 2] = log_size_lo + bid_scale * (log_size_hi - log_size_lo)
            actions[:, 3] = log_size_lo + ask_scale * (log_size_hi - log_size_lo)
        return actions

    def _hybrid_action(self, raw_action, mid, inventory, t_remaining, sigma_hat):
        if self.backbone is None:
            raise ValueError("Hybrid mode requires an Avellaneda-Stoikov backbone.")

        raw_action = np.atleast_2d(raw_action)
        mid_arr = np.atleast_1d(mid)
        inv_arr = np.atleast_1d(inventory)
        t_arr = np.atleast_1d(t_remaining)
        sig_arr = np.atleast_1d(sigma_hat)
        n = raw_action.shape[0]
        tick = 0.01
        log_size_lo = np.log(0.1)
        log_size_hi = np.log(2.0)

        actions = np.zeros((n, self.cfg.action_dim), dtype=np.float32)
        for i in range(n):
            sigma = max(float(sig_arr[i]), 1e-6)
            tau = max(float(t_arr[i]), 0.0) * self.backbone.horizon
            gamma_eff = max(self.backbone.gamma, 1e-5)
            delta_as = max(
                gamma_eff * sigma ** 2 * tau
                + (2.0 / gamma_eff) * np.log(1.0 + gamma_eff / self.backbone.k),
                tick,
            )
            spread_scale = np.exp(0.20 * float(raw_action[i, 0]))
            delta_hybrid = float(np.clip(delta_as * spread_scale, tick, 0.20))  # 0.20=full spread上限，half=0.10，fill~16%

            reservation = float(mid_arr[i]) - float(inv_arr[i]) * gamma_eff * sigma ** 2 * tau
            skew = float(raw_action[i, 1]) * 0.20 * delta_hybrid
            center = reservation + skew

            bid_off = max(float(mid_arr[i]) - (center - delta_hybrid / 2.0), tick)
            ask_off = max((center + delta_hybrid / 2.0) - float(mid_arr[i]), tick)
            bid_off, ask_off, bid_sz_scale, ask_sz_scale = self._inventory_risk_overlay(
                np.array([bid_off], dtype=np.float32),
                np.array([ask_off], dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.array([float(inv_arr[i])], dtype=np.float32),
            )

            actions[i, 0] = np.log(max(float(bid_off[0]), tick))
            actions[i, 1] = np.log(max(float(ask_off[0]), tick))
            if self.cfg.action_dim > 2:
                bid_scale = (float(raw_action[i, 2]) + 1.0) * 0.5
                ask_scale = (float(raw_action[i, 3]) + 1.0) * 0.5
                actions[i, 2] = log_size_lo + np.clip(bid_scale * float(bid_sz_scale[0]), 0.0, 1.0) * (log_size_hi - log_size_lo)
                actions[i, 3] = log_size_lo + np.clip(ask_scale * float(ask_sz_scale[0]), 0.0, 1.0) * (log_size_hi - log_size_lo)
        return actions

    def _inventory_risk_overlay(self, bid_off, ask_off, bid_sz_scale, ask_sz_scale, inventory):
        bid_off = np.asarray(bid_off, dtype=np.float32).copy()
        ask_off = np.asarray(ask_off, dtype=np.float32).copy()
        bid_sz_scale = np.asarray(bid_sz_scale, dtype=np.float32).copy()
        ask_sz_scale = np.asarray(ask_sz_scale, dtype=np.float32).copy()
        inv_frac = np.clip(
            np.asarray(inventory, dtype=np.float32) / max(self.cfg.inventory_limit, 1e-8),
            -1.0,
            1.0,
        )
        excess = np.maximum(np.abs(inv_frac) - 0.50, 0.0)
        strength = 1.0 + 1.5 * excess / 0.50
        long_mask = inv_frac > 0
        short_mask = inv_frac < 0

        bid_off[long_mask] *= strength[long_mask]
        ask_off[long_mask] = np.maximum(ask_off[long_mask] / strength[long_mask], 0.01)
        bid_sz_scale[long_mask] *= np.maximum(0.15, 1.0 / (strength[long_mask] ** 2))
        ask_sz_scale[long_mask] *= np.minimum(strength[long_mask], 1.5)

        ask_off[short_mask] *= strength[short_mask]
        bid_off[short_mask] = np.maximum(bid_off[short_mask] / strength[short_mask], 0.01)
        ask_sz_scale[short_mask] *= np.maximum(0.15, 1.0 / (strength[short_mask] ** 2))
        bid_sz_scale[short_mask] *= np.minimum(strength[short_mask], 1.5)
        return bid_off, ask_off, bid_sz_scale, ask_sz_scale

    def update(self, last_obs, last_done):
        with torch.no_grad():
            obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.dev)
            _, _, last_values = self.ac(obs_t)
            last_value = last_values.squeeze(-1).cpu()
            last_done_t = torch.as_tensor(last_done, dtype=torch.float32)

        self.buffer.compute_gae(last_value, last_done_t)
        stats = {name: [] for name in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]}

        early_stop = False
        for _ in range(self.cfg.n_epochs):
            if early_stop:
                break
            for ob, ac, old_lp, adv, ret, _cost in self.buffer.get_batches(self.cfg.batch_size, self.eta):
                _, new_lp, ent, new_val = self.ac.get_action_and_value(ob, ac)
                ratio = (new_lp - old_lp).exp()

                surrogate_1 = ratio * adv
                surrogate_2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = 0.5 * (new_val - ret).pow(2).mean()
                entropy_loss = -ent.mean()
                loss = policy_loss + self.cfg.vf_coef * value_loss + self.cfg.entropy_coef * entropy_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_lp - old_lp)).mean()
                    clip_frac = ((ratio - 1).abs() > self.cfg.clip_eps).float().mean()

                kl_val = approx_kl.item()
                if not np.isfinite(kl_val):
                    kl_val = 0.0

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(-entropy_loss.item())
                stats["approx_kl"].append(kl_val)
                stats["clip_frac"].append(clip_frac.item())

                # target_kl 早停：梯度步之后检查，stats 已记录不会为空
                if self.cfg.target_kl is not None and kl_val > 1.5 * self.cfg.target_kl:
                    early_stop = True
                    break

        avg_cost = self.buffer.costs[: self.buffer.ptr].mean().item()
        if self.cfg.use_lagrangian:
            self.eta = min(
                max(0.0, self.eta + self.cfg.eta_lr * (avg_cost - self.cfg.constraint_budget)),
                self.cfg.eta_max,
            )

        self.sched.step()
        self.buffer.reset()
        result = {k: float(np.mean(v)) for k, v in stats.items()}
        result["eta"] = self.eta
        result["constraint"] = avg_cost
        return result

    def save(self, path):
        torch.save(
            {
                "ac": self.ac.state_dict(),
                "opt": self.opt.state_dict(),
                "eta": self.eta,
                "step": self.global_step,
                "policy_mode": self.cfg.policy_mode,
            },
            path,
        )
        print(f"  Saved -> {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.dev)
        self.ac.load_state_dict(ckpt["ac"])
        self.opt.load_state_dict(ckpt["opt"])
        self.eta = ckpt["eta"]
        self.global_step = ckpt["step"]
        print(f"  Loaded <- {path}")
