"""Uniform critic-value access for PPO and vendored CleanRL TD3 models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def critic_values(model: Any, observations: np.ndarray) -> np.ndarray:
    """Evaluate the base policy's critic and return one value per observation."""

    observations = np.asarray(observations)
    is_single = observations.ndim == 1
    batch = observations.reshape(1, -1) if is_single else observations
    tensor_obs = torch.as_tensor(batch, dtype=torch.float32, device=model.device)

    with torch.no_grad():
        policy = getattr(model, "policy", None)
        if policy is not None and hasattr(policy, "predict_values"):
            values = policy.predict_values(tensor_obs)
        elif hasattr(model, "actor") and hasattr(model, "critic"):
            actions = model.actor(tensor_obs)
            twin_values = model.critic(tensor_obs, actions)
            values = torch.min(torch.cat(twin_values, dim=1), dim=1, keepdim=True)[0]
        else:
            raise TypeError(
                "critic evaluation requires either a state-value policy or an "
                "actor with twin action-value critics"
            )

    result = values.detach().cpu().numpy().reshape(-1)
    return result[0] if is_single else result
