import gymnasium as gym
import numpy as np

from calfwrapper.fallback.base import FallbackPolicy


class CartPoleFallbackPolicy(FallbackPolicy):
    def __init__(
        self,
        pd_coefs: list[float],
        gain: float,
        gain_pos_vel: float,
        action_min: float,
        action_max: float,
        env_id: str = "CALFWrapper/CartPoleSwingUpLong-v0",
        gain_pos: float = 0.0,
        swing_position_reference_gain: float = 0.0,
        switch_loc: float = 0.9,
        blend_width: float = 0.0,
        velocity_brake_threshold: float | None = None,
        velocity_brake_position_threshold: float | None = None,
        action_bias: float = 0.0,
    ):
        self.pd_coefs = pd_coefs
        self.gain = gain
        self.gain_pos_vel = gain_pos_vel
        self.gain_pos = gain_pos
        self.swing_position_reference_gain = swing_position_reference_gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.blend_width = blend_width
        if velocity_brake_threshold is not None and velocity_brake_threshold <= 0:
            raise ValueError("velocity_brake_threshold must be positive")
        if velocity_brake_position_threshold is not None and velocity_brake_position_threshold < 0:
            raise ValueError("velocity_brake_position_threshold must be nonnegative")
        self.velocity_brake_threshold = velocity_brake_threshold
        self.velocity_brake_position_threshold = velocity_brake_position_threshold
        self.action_bias = action_bias

        env = gym.make(env_id)
        self.masscart = env.unwrapped.masscart
        self.masspole = env.unwrapped.masspole
        self.gravconst = env.unwrapped.gravconst
        self.length = env.unwrapped.length
        self.total_mass = self.masscart + self.masspole
        self.moment_of_inertia = 4 / 3 * self.masspole * self.length**2
        self.polemass_length = self.masspole * self.length
        env.close()

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
            0.5 * self.moment_of_inertia * angle_vel** 2
            + self.polemass_length * self.gravconst * (cos_angle - 1)
        )
        swing_position_reference = self.swing_position_reference_gain * sin_angle
        target_acc = self.gain * (
            energy * cos_angle * angle_vel - self.gain_pos_vel * pos_vel
        ) - self.gain_pos * (pos - swing_position_reference)

        energy_based_action = (
            self.total_mass * target_acc
            - self.polemass_length * sin_angle * angle_vel**2
            + cos_angle
            * self.polemass_length
            * (self.gravconst * sin_angle - target_acc)
            / self.moment_of_inertia
        )

        pd_action = (
            angle * self.pd_coefs[0]
            + pos * self.pd_coefs[1]
            + angle_vel * self.pd_coefs[2]
            + pos_vel * self.pd_coefs[3]
        )
        if self.blend_width > 0:
            blend = np.clip(
                (cos_angle - (self.switch_loc - self.blend_width)) / (2 * self.blend_width),
                0.0,
                1.0,
            )
            blend = blend**2 * (3 - 2 * blend)
            action = (1 - blend) * energy_based_action + blend * pd_action
        else:
            action = np.where(cos_angle > self.switch_loc, pd_action, energy_based_action)
        if self.velocity_brake_threshold is not None:
            braking_action = np.where(pos_vel > 0, self.action_min, self.action_max)
            braking_required = np.abs(pos_vel) > self.velocity_brake_threshold
            if self.velocity_brake_position_threshold is not None:
                braking_required &= np.abs(pos) > self.velocity_brake_position_threshold
            action = np.where(
                braking_required,
                braking_action,
                action,
            )
        action = np.clip(action + self.action_bias, self.action_min, self.action_max)

        return action
