import math
import numpy as np
from src.envs.underwaterdrone import TOP_Y, DRONE_MASS, GRAVITY, MAX_F_LONG, MAX_F_LAT
from src.envs.robot_dynamics import RobotDynamicsConfig
from src.envs.robot_navigation import RobotNavigationConfig
from src.envs.robot_navigation_const_speed import RobotNavigationConstSpeedConfig


class UnderwaterDroneNominalController:
    def __init__(
        self,
        kp_y: float = 2.0,
        kd_y: float = 1.2,
        kp_x: float = 1.5,
        kd_x: float = 0.8,
    ) -> None:
        self.kp_y = kp_y
        self.kd_y = kd_y
        self.kp_x = kp_x
        self.kd_x = kd_x

    def get_action(self, obs):
        if len(obs.shape) == 1:
            x, y, cos_theta, sin_theta, v_x, v_y, _ = obs
        else:  # obs is a batch of observations
            x, y, cos_theta, sin_theta, v_x, v_y = (
                obs[:, i, np.newaxis] for i in range(6)
            )

        y_err = TOP_Y - y
        Fy = GRAVITY * DRONE_MASS + self.kp_y * y_err - self.kd_y * v_y

        x_ref = 0.0
        x_err = x_ref - x
        Fx = self.kp_x * x_err - self.kd_x * v_x

        F_long = cos_theta * Fx + sin_theta * Fy
        F_lat = -sin_theta * Fx + cos_theta * Fy

        F_long = np.clip(F_long, -MAX_F_LONG, MAX_F_LONG)
        F_lat = np.clip(F_lat, -MAX_F_LAT, MAX_F_LAT)

        return np.hstack([F_long, F_lat])


class LidarNavController:
    def __init__(self, kp_angle=2.0):
        from src.envs.lidarnav import LidarNavEnv

        env = LidarNavEnv()
        self.goal_pos = env.goal_pos
        self.max_velocity = env.max_velocity
        self.max_angular_velocity = env.max_angular_velocity
        self.kp_angle = kp_angle

    def get_action(self, obs):
        robot_pos = obs.reshape(-1)[:2]
        to_goal = self.goal_pos - robot_pos
        angle_to_goal = np.arctan2(to_goal[1], to_goal[0])
        robot_angle = np.arctan2(obs.reshape(-1)[2], obs.reshape(-1)[3])
        angle_error = (angle_to_goal - robot_angle) % (2 * np.pi)
        if angle_error > np.pi:
            angle_error -= 2 * np.pi

        # Simple P controller for angular velocity
        omega = self.kp_angle * angle_error
        omega = np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity)
        v = self.max_velocity * np.cos(angle_error)
        v = max(0.1, v)
        return np.array([[v, omega]])


class RobotNavigationGoalController:
    def __init__(
        self,
        max_speed: float | None = None,
        turn_gain: float = 1.0,
        max_turn_rate: float = math.pi,
        cruise_speed: float = 0.1,
    ) -> None:
        config = RobotNavigationConfig()
        self.max_speed = float(config.max_speed if max_speed is None else max_speed)
        self.turn_gain = float(turn_gain)
        self.max_turn_rate = float(max_turn_rate)
        if cruise_speed is None:
            self.cruise_speed = self.max_speed
        else:
            self.cruise_speed = float(cruise_speed)
        self.cruise_speed = max(0.0, min(self.cruise_speed, self.max_speed))

    def get_action(self, obs):
        obs = np.asarray(obs)
        batched = obs.ndim > 1
        obs = obs if batched else obs[None, :]

        robot = obs[:, 0:2]
        heading_cos = obs[:, 2:3]
        heading_sin = obs[:, 3:4]
        current_angle = np.arctan2(heading_sin, heading_cos)
        goal = obs[:, 4:6]

        delta = goal - robot
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        desired_angle = np.arctan2(delta[:, 1], delta[:, 0])[:, None]
        desired_angle = np.where(distance < 1e-8, current_angle, desired_angle)

        angle_error = desired_angle - current_angle
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        angular_velocity = np.clip(
            self.turn_gain * angle_error, -self.max_turn_rate, self.max_turn_rate
        )

        if self.max_speed <= 0.0 or self.cruise_speed <= 0.0:
            speed = np.zeros_like(distance)
        else:
            speed = np.clip(distance / (self.max_speed * 8.0), 0.0, 1.0)
            speed = np.minimum(speed, self.cruise_speed)
        speed = np.where(distance < 1e-8, 0.0, speed)

        actions = np.hstack([speed, angular_velocity]).astype(np.float32)
        return actions if batched else actions[0]


class RobotDynamicsCollectController:
    def __init__(
        self,
        max_speed: float | None = None,
        turn_gain: float = 1.0,
        max_turn_rate: float | None = None,
        cruise_speed: float = 0.1,
    ) -> None:
        config = RobotDynamicsConfig()
        self.max_speed = float(config.max_speed if max_speed is None else max_speed)
        self.turn_gain = float(turn_gain)
        if max_turn_rate is None:
            self.max_turn_rate = float(config.max_angular_velocity)
        else:
            self.max_turn_rate = float(max_turn_rate)
        if cruise_speed is None:
            self.cruise_speed = self.max_speed
        else:
            self.cruise_speed = float(cruise_speed)
        self.cruise_speed = max(0.0, min(self.cruise_speed, self.max_speed))

    def get_action(self, obs):
        obs = np.asarray(obs)
        batched = obs.ndim > 1
        obs = obs if batched else obs[None, :]
        batch = obs.shape[0]

        robot_to_target = -obs[:, 0:2]
        heading_cos = obs[:, 2:3]
        heading_sin = obs[:, 3:4]
        current_angle = np.arctan2(heading_sin, heading_cos)

        vec = robot_to_target
        n_collectables = max(0, (obs.shape[1] - 4) // 3)
        if n_collectables > 0:
            collectable_block = obs[:, 4:].reshape(batch, n_collectables, 3)
            collectable_vecs = -collectable_block[:, :, 0:2]
            captured = collectable_block[:, :, 2] > 0.5
            distances = np.linalg.norm(collectable_vecs, axis=2)
            distances = np.where(captured, np.inf, distances)
            nearest_idx = np.argmin(distances, axis=1)
            nearest_dist = distances[np.arange(batch), nearest_idx]
            has_target = np.isfinite(nearest_dist)
            nearest_vec = collectable_vecs[np.arange(batch), nearest_idx]
            vec = np.where(has_target[:, None], nearest_vec, robot_to_target)

        distance = np.linalg.norm(vec, axis=1, keepdims=True)
        desired_angle = np.arctan2(vec[:, 1], vec[:, 0])[:, None]
        desired_angle = np.where(distance < 1e-8, current_angle, desired_angle)

        angle_error = desired_angle - current_angle
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        angular_velocity = np.clip(
            self.turn_gain * angle_error, -self.max_turn_rate, self.max_turn_rate
        )

        if self.max_speed <= 0.0 or self.cruise_speed <= 0.0:
            speed = np.zeros_like(distance)
        else:
            speed = np.clip(distance / (self.max_speed * 8.0), 0.0, 1.0)
            speed = np.minimum(speed, self.cruise_speed)
        speed = np.where(distance < 1e-8, 0.0, speed)

        actions = np.hstack([speed, angular_velocity]).astype(np.float32)
        return actions if batched else actions[0]


class RobotDynamicsGoalController:
    def __init__(
        self,
        max_speed: float | None = None,
        turn_gain: float = 1.0,
        max_turn_rate: float | None = None,
        cruise_speed: float = 0.1,
    ) -> None:
        config = RobotDynamicsConfig()
        self.max_speed = float(config.max_speed if max_speed is None else max_speed)
        self.turn_gain = float(turn_gain)
        if max_turn_rate is None:
            self.max_turn_rate = float(config.max_angular_velocity)
        else:
            self.max_turn_rate = float(max_turn_rate)
        if cruise_speed is None:
            self.cruise_speed = self.max_speed
        else:
            self.cruise_speed = float(cruise_speed)
        self.cruise_speed = max(0.0, min(self.cruise_speed, self.max_speed))

    def get_action(self, obs):
        obs = np.asarray(obs)
        batched = obs.ndim > 1
        obs = obs if batched else obs[None, :]

        vec = -obs[:, 0:2]
        heading_cos = obs[:, 2:3]
        heading_sin = obs[:, 3:4]
        current_angle = np.arctan2(heading_sin, heading_cos)

        distance = np.linalg.norm(vec, axis=1, keepdims=True)
        desired_angle = np.arctan2(vec[:, 1], vec[:, 0])[:, None]
        desired_angle = np.where(distance < 1e-8, current_angle, desired_angle)

        angle_error = desired_angle - current_angle
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        angular_velocity = np.clip(
            self.turn_gain * angle_error, -self.max_turn_rate, self.max_turn_rate
        )

        if self.max_speed <= 0.0 or self.cruise_speed <= 0.0:
            speed = np.zeros_like(distance)
        else:
            speed = np.clip(distance / (self.max_speed * 8.0), 0.0, 1.0)
            speed = np.minimum(speed, self.cruise_speed)
        speed = np.where(distance < 1e-8, 0.0, speed)

        actions = np.hstack([speed, angular_velocity]).astype(np.float32)
        return actions if batched else actions[0]


class RobotNavigationConstSpeedGoalController:
    """Controller for const-speed env. Returns 1D action [angular_velocity]."""

    def __init__(
        self,
        turn_gain: float = 1.0,
        max_turn_rate: float | None = None,
    ) -> None:
        config = RobotNavigationConstSpeedConfig()
        self.turn_gain = float(turn_gain)
        self.max_turn_rate = (
            float(config.max_angular_velocity) if max_turn_rate is None else float(max_turn_rate)
        )

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        batched = obs.ndim > 1
        obs = obs if batched else obs[None, :]

        robot = obs[:, 0:2]
        heading_cos = obs[:, 2:3]
        heading_sin = obs[:, 3:4]
        current_angle = np.arctan2(heading_sin, heading_cos)
        goal = obs[:, 4:6]

        delta = goal - robot
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        desired_angle = np.arctan2(delta[:, 1], delta[:, 0])[:, None]
        desired_angle = np.where(distance < 1e-8, current_angle, desired_angle)

        angle_error = desired_angle - current_angle
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        angular_velocity = np.clip(
            self.turn_gain * angle_error, -self.max_turn_rate, self.max_turn_rate
        )

        actions = angular_velocity.astype(np.float32)
        return actions if batched else actions[0]
