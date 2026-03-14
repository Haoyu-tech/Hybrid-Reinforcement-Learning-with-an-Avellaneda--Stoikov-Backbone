from typing import List, Any, Type, Optional, Union, Sequence

import gym
import gymnasium.spaces as gymn_spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

from mbt_gym.gym.TradingEnvironment import TradingEnvironment


def _to_gymnasium_space(space):
    """Convert classic gym spaces to gymnasium spaces for SB3 compatibility."""
    if isinstance(space, gymn_spaces.Space):
        return space
    if isinstance(space, gym.spaces.Box):
        return gymn_spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        return gymn_spaces.Discrete(n=space.n)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return gymn_spaces.MultiDiscrete(nvec=space.nvec)
    if isinstance(space, gym.spaces.MultiBinary):
        return gymn_spaces.MultiBinary(n=space.n)
    if isinstance(space, gym.spaces.Tuple):
        return gymn_spaces.Tuple(tuple(_to_gymnasium_space(s) for s in space.spaces))
    if isinstance(space, gym.spaces.Dict):
        return gymn_spaces.Dict({k: _to_gymnasium_space(v) for k, v in space.spaces.items()})
    return space


class StableBaselinesTradingEnvironment(VecEnv):
    def __init__(
        self,
        trading_env: TradingEnvironment,
        store_terminal_observation_info: bool = True,
    ):
        self.env = trading_env
        self.store_terminal_observation_info = store_terminal_observation_info
        self.actions: np.ndarray = self.env.action_space.sample()
        observation_space = _to_gymnasium_space(self.env.observation_space)
        action_space = _to_gymnasium_space(self.env.action_space)
        super().__init__(self.env.num_trajectories, observation_space, action_space)

    def reset(self) -> VecEnvObs:
        return self.env.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.env.step(self.actions)
        if dones.min():
            if self.store_terminal_observation_info:
                infos = infos.copy()
                for count, info in enumerate(infos):
                    # save final observation where user can get it, then automatically reset (an SB3 convention).
                    info["terminal_observation"] = obs[count, :]
            obs = self.env.reset()
        return obs, rewards, dones, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        target_indices = list(self._get_indices(indices))
        if attr_name == "render_mode":
            value = getattr(self.env, "render_mode", None)
            return [value for _ in target_indices]
        if not hasattr(self.env, attr_name):
            raise AttributeError(f"Underlying environment has no attribute '{attr_name}'")
        value = getattr(self.env, attr_name)
        return [value for _ in target_indices]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_indices = list(self._get_indices(indices))
        if len(target_indices) == 0:
            return
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        target_indices = list(self._get_indices(indices))
        if not hasattr(self.env, method_name):
            raise AttributeError(f"Underlying environment has no method '{method_name}'")
        method = getattr(self.env, method_name)
        result = method(*method_args, **method_kwargs)
        return [result for _ in target_indices]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in range(self.env.num_trajectories)]

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self.env.seed(seed)

    def get_images(self) -> Sequence[np.ndarray]:
        return [None for _ in range(self.env.num_trajectories)]

    @property
    def num_trajectories(self):
        return self.env.num_trajectories

    @property
    def n_steps(self):
        return self.env.n_steps
