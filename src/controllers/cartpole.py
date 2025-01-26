import gymnasium as gym
import numpy as np
from src.controllers.controller import Controller


class CartpoleEnergyBasedStabilizingPolicy(Controller):
    def __init__(
        self,
        pd_coefs: list[float],
        gain: float,
        gain_pos_vel: float,
        action_min: float,
        action_max: float,
        env_id: str = "CartpoleSwingupEnvLong-v0",
    ):
        self.pd_coefs = pd_coefs
        self.gain = gain
        self.gain_pos_vel = gain_pos_vel
        self.action_min = action_min
        self.action_max = action_max

        env = gym.make(env_id)
        self.masscart = env.unwrapped.masscart
        self.masspole = env.unwrapped.masspole
        self.gravconst = env.unwrapped.gravconst
        self.length = env.unwrapped.length
        self.total_mass = self.masscart + self.masspole
        self.moment_of_inertia = 4 / 3 * self.masspole * self.length**2
        self.polemass_length = self.masspole * self.length

    def get_action(self, observation):
        if observation.ndim == 1:
            pos = observation[np.newaxis, 0]
            pos_vel = observation[np.newaxis, 1]
            cos_angle = observation[np.newaxis, 2]
            sin_angle = observation[np.newaxis, 3]
            angle = np.arctan2(sin_angle, cos_angle)
            angle_vel = observation[np.newaxis, 4]
        else:
            pos = observation[:, np.newaxis, 0]
            pos_vel = observation[:, np.newaxis, 1]
            cos_angle = observation[:, np.newaxis, 2]
            sin_angle = observation[:, np.newaxis, 3]
            angle = np.arctan2(sin_angle, cos_angle)
            angle_vel = observation[:, np.newaxis, 4]

        energy = (
            0.5 * self.moment_of_inertia * angle_vel**2
            + self.polemass_length * self.gravconst * (cos_angle - 1)
        )
        target_acc = self.gain * (
            energy * cos_angle * angle_vel - self.gain_pos_vel * pos_vel
        )

        energy_based_action = (
            self.total_mass * target_acc
            - self.polemass_length * sin_angle * angle_vel
            + cos_angle
            * self.polemass_length
            * (self.gravconst * sin_angle - target_acc)
            / self.moment_of_inertia
        )

        pos_clipped = np.clip(pos, -1.0, 1.0)
        pos_vel_clipped = np.clip(pos_vel, -1.0, 1.0)

        pd_action = (
            angle * self.pd_coefs[0]
            + pos_clipped * self.pd_coefs[1]
            + angle_vel * self.pd_coefs[2]
            + pos_vel_clipped * self.pd_coefs[3]
        )
        action = np.where(cos_angle > 0.9, pd_action, energy_based_action)
        action = np.clip(action, self.action_min, self.action_max)

        return action
