import numpy as np
import gymnasium as gym
from src.controllers.controller import Controller


class EnergyBasedStabilizingPolicy(Controller):
    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        switch_loc: float,
        switch_vel_loc: float,
        pd_coeffs: list[float],
        env_id: str = "Pendulum-v1",
    ):
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.switch_vel_loc = switch_vel_loc

        self.mass = gym.make(env_id).unwrapped.m
        self.length = gym.make(env_id).unwrapped.l
        self.grav_const = gym.make(env_id).unwrapped.g

        self.pendulum_moment_inertia = self.mass * self.length**2 / 3

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        if observation.ndim == 1:
            cos_angle = observation[np.newaxis, 0]
            sin_angle = observation[np.newaxis, 1]
            angle_vel = observation[np.newaxis, 2]
        else:
            cos_angle = observation[:, np.newaxis, 0]
            sin_angle = observation[:, np.newaxis, 1]
            angle_vel = observation[:, np.newaxis, 2]

        energy_total = (
            self.mass * self.grav_const * self.length * (cos_angle - 1) / 2
            + 1 / 2 * self.pendulum_moment_inertia * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(angle_vel * energy_total)
        pd_control_action = (
            -self.pd_coeffs[0] * sin_angle - self.pd_coeffs[1] * angle_vel
        )

        action = np.clip(
            np.where(
                (cos_angle <= self.switch_loc)
                | (np.abs(angle_vel) > self.switch_vel_loc),
                energy_control_action,
                pd_control_action,
            ),
            self.action_min,
            self.action_max,
        )

        return action

