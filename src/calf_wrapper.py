from gymnasium import Wrapper
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from src.controllers.controller import Controller
from typing import Any, Callable, Optional, Union

from src.critic_values import critic_values


class CALFWrapper(Wrapper):
    def __init__(
        self,
        env: VecEnv,
        model: Any,
        stabilizing_policy: Controller,
        calf_change_rate=0.01,
        relaxprob_init=0.5,
        relaxprob_factor=1.0,
        seed: Optional[int] = None,
        critic_upper_bound: Optional[float] = None,
        fallback_lock_mask: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(env)
        self.model = model
        self.calf_change_rate = calf_change_rate
        self.relaxprob_init = relaxprob_init
        self.relaxprob_factor = relaxprob_factor
        self.stabilizing_policy = stabilizing_policy
        self.critic_upper_bound = critic_upper_bound
        self.fallback_lock_mask = fallback_lock_mask
        self.relaxprob = float(self.relaxprob_init)
        self.np_rng = np.random.default_rng(seed=seed)

    def value(self, obs: np.ndarray) -> Union[float, np.ndarray]:
        values = critic_values(self.model, obs)
        if self.critic_upper_bound is not None:
            values = np.minimum(values, self.critic_upper_bound)
        if np.ndim(values) == 0:
            return np.asarray(values).reshape(-1)[0]
        return np.asarray(values).reshape(-1, 1)

    def step(self, base_action: np.ndarray):
        value = self.value(self.obs)
        value_decay = value - self.best_value - self.calf_change_rate
        fallback_locked = np.zeros_like(value_decay, dtype=bool)
        if self.fallback_lock_mask is not None:
            fallback_locked = np.asarray(
                self.fallback_lock_mask(self.obs), dtype=bool
            ).reshape(value_decay.shape)
        deterministic_acceptance = (value_decay >= 0) & ~fallback_locked
        self.best_value = np.where(deterministic_acceptance, value, self.best_value)

        probabilistic_acceptance = (
            self.np_rng.random(size=value_decay.shape) < self.relaxprob
        ) & ~fallback_locked
        is_base_action_applied = deterministic_acceptance | probabilistic_acceptance
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
                    "calf.decay_happened": deterministic_acceptance[i, 0],
                    "calf.deterministic_acceptance": deterministic_acceptance[i, 0],
                    "calf.probabilistic_acceptance": probabilistic_acceptance[i, 0],
                    "calf.base_action_applied": is_base_action_applied[i, 0],
                    "calf.fallback_locked": fallback_locked[i, 0],
                    "calf.action": action[i, :],
                }
        else:  # single env
            info |= {
                "calf.relaxprob": np.copy(self.relaxprob),
                "calf.decay_happened": deterministic_acceptance,
                "calf.deterministic_acceptance": deterministic_acceptance,
                "calf.probabilistic_acceptance": probabilistic_acceptance,
                "calf.base_action_applied": is_base_action_applied,
                "calf.fallback_locked": fallback_locked,
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
