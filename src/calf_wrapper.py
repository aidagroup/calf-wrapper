from gymnasium import Wrapper
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO
from src.controllers.controller import Controller
from typing import Optional, Union


class CALFWrapper(Wrapper):
    def __init__(
        self,
        env: VecEnv,
        model: PPO,
        stabilizing_policy: Controller,
        calf_change_rate=0.01,
        relaxprob_init=0.5,
        relaxprob_factor=1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.model = model
        self.calf_change_rate = calf_change_rate
        self.relaxprob_init = relaxprob_init
        self.relaxprob_factor = relaxprob_factor
        self.stabilizing_policy = stabilizing_policy
        self.relaxprob = float(self.relaxprob_init)
        self.np_rng = np.random.default_rng(seed=seed)

    def value(self, obs: np.ndarray) -> Union[float, np.ndarray]:
        with torch.no_grad():
            if obs.ndim == 1:
                tensor_obs = torch.tensor(obs.reshape(1, -1), device=self.model.device)
            else:
                tensor_obs = torch.tensor(obs, device=self.model.device)

            values = self.model.policy.predict_values(tensor_obs).cpu().numpy()

            if obs.ndim == 1:
                return values[0][0]
            else:
                return values

    def step(self, base_action: np.ndarray):
        value = self.value(self.obs)
        value_decay = value - self.best_value - self.calf_change_rate
        self.best_value = np.where(value_decay >= 0, value, self.best_value)

        is_base_action_applied = (value_decay >= 0) | (
            self.np_rng.random(size=value_decay.shape) < self.relaxprob
        )
        action = np.where(
            is_base_action_applied,
            base_action,
            self.stabilizing_policy.get_action(self.obs),
        )
        env_step_output = list(self.env.step(action))
        next_obs, info = env_step_output[0], env_step_output[-1]
        self.obs = np.copy(next_obs)

        if isinstance(info, list):  # vectorized env
            for i in range(len(info)):
                info[i] |= {
                    "calf.relaxprob": np.copy(self.relaxprob),
                    "calf.decay_happened": (value_decay >= 0)[i, 0],
                    "calf.base_action_applied": is_base_action_applied[i, 0],
                    "calf.action": action[i, :],
                }
        else:  # single env
            info |= {
                "calf.relaxprob": np.copy(self.relaxprob),
                "calf.decay_happened": value_decay >= 0,
                "calf.base_action_applied": is_base_action_applied,
                "calf.action": action,
            }
        env_step_output[-1] = info

        self.relaxprob *= self.relaxprob_factor
        return tuple(env_step_output)

    def reset(self, *args, **kwargs):
        self.relaxprob = float(self.relaxprob_init)
        reset_output = self.env.reset(*args, **kwargs)
        if isinstance(reset_output, tuple):
            self.obs = reset_output[0]
        else:
            self.obs = reset_output
        self.best_value = self.value(self.obs)
        return np.copy(self.obs)
