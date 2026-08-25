from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from src.utils.metrics_controller import MetricsCollector


@dataclass
class RobotDynamicsConfig:
    """Configuration values for the robot dynamics environment."""

    max_speed: float = 0.15
    control_dt: float = 0.05
    max_angular_velocity: float = math.pi
    world_low: float = 0.0
    world_high: float = 1.0
    start_position_distribution: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (0.5, 0.1),
        (0.9, 0.9),
    )
    start_angle_distribution: Tuple[float, float] = (0.0, 2 * math.pi)
    target_position: Tuple[float, float] = (0, 0.5)
    target_radius: float = 0.05
    n_collectables: int = 4
    collectable_radius: float = 0.1
    collectable_speed_range: Tuple[float, float] = (0.08, 0.18)
    collectable_speed: Optional[float] = None
    collectable_noise_std: float = 0.25
    collectable_reward: float = 50.0
    terminal_reward: float = 10.0
    window_size: Tuple[int, int] = (800, 600)


class RobotDynamicsEnv(gym.Env[np.ndarray, np.ndarray]):
    """Simple 2D robot dynamics environment with optional pygame rendering."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        config: Optional[RobotDynamicsConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(
                "Unsupported render_mode. Use None, 'human', or 'rgb_array'."
            )
        self.render_mode = render_mode
        self.config = config or RobotDynamicsConfig()
        if self.config.control_dt <= 0.0:
            raise ValueError("RobotDynamicsEnv control_dt must be positive.")

        self._rng: np.random.Generator
        self._steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)
        self.robot_position = np.zeros(2, dtype=np.float32)
        self.robot_angle = 0.0
        self.target_position = np.array(self.config.target_position, dtype=np.float32)
        self.target_radius = self.config.target_radius
        self.collectable_positions = np.zeros(
            (self.config.n_collectables, 2), dtype=np.float32
        )
        self.collectable_captured_mask = [False] * self.config.n_collectables
        self._collectable_velocities = np.zeros(
            (self.config.n_collectables, 2), dtype=np.float32
        )
        self._collectable_speed_scales = np.zeros(
            self.config.n_collectables, dtype=np.float32
        )
        self._pygame = None
        self._window = None
        self._clock = None
        self._surface = None

        self.action_space = spaces.Box(
            low=np.array(
                [-self.config.max_speed, -self.config.max_angular_velocity],
                dtype=np.float32,
            ),
            high=np.array(
                [self.config.max_speed, self.config.max_angular_velocity],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        # Observation: [robot_x, robot_y, cos(angle), sin(angle), collectable_x, collectable_y, collectable_captured]
        obs_low = np.array(
            [
                self.config.world_low - self.config.world_high,
                self.config.world_low - self.config.world_high,
                -1.0,
                -1.0,
            ]
            + [
                self.config.world_low - self.config.world_high,
                self.config.world_low - self.config.world_high,
                0.0,
            ]
            * self.config.n_collectables,
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                self.config.world_high - self.config.world_low,
                self.config.world_high - self.config.world_low,
                1.0,
                1.0,
            ]
            + [
                self.config.world_high - self.config.world_low,
                self.config.world_high - self.config.world_low,
                1.0,
            ]
            * self.config.n_collectables,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        if seed is None:
            self.np_random, _ = seeding.np_random(None)
        else:
            self.np_random, _ = seeding.np_random(seed)
        self._rng = self.np_random

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self._rng = self.np_random

        self._steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)

        position = self._rng.uniform(
            self.config.start_position_distribution[0],
            self.config.start_position_distribution[1],
            2,
        )
        angle = self._rng.uniform(
            self.config.start_angle_distribution[0],
            self.config.start_angle_distribution[1],
        )

        self.robot_position = position
        self.robot_angle = self._wrap_angle(angle)

        # Spawn collectable at random position within world bounds
        self.collectable_positions = self._rng.uniform(
            self.config.world_low,
            self.config.world_high,
            (self.config.n_collectables, 2),
        ).astype(np.float32)
        self.collectable_captured = [False] * self.config.n_collectables
        self._initialize_collectable_motion()

        observation = self._get_observation()
        info = {
            "distance_to_target": np.linalg.norm(
                self.robot_position - self.target_position
            ),
            "collectable_captured": sum(self.collectable_captured),
        }
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._last_action = action
        self._steps += 1
        self._update_collectable_positions()

        speed = float(action[0])
        angular_velocity = float(action[1])

        delta = np.array(
            [math.cos(self.robot_angle), math.sin(self.robot_angle)],
            dtype=np.float32,
        )
        self.robot_position = self.robot_position + delta * (
            speed * self.config.control_dt
        )
        self.robot_position = np.clip(
            self.robot_position,
            self.config.world_low,
            self.config.world_high,
        )
        self.robot_angle = self._wrap_angle(
            self.robot_angle + angular_velocity * self.config.control_dt
        )

        # Check if collectable is captured
        collectable_reward = 0.0
        for i in range(self.config.n_collectables):
            if not self.collectable_captured[i]:
                distance_to_collectable = np.linalg.norm(
                    self.robot_position - self.collectable_positions[i]
                )
                if distance_to_collectable < self.config.collectable_radius:
                    collectable_reward = self.config.collectable_reward
                    self.collectable_captured[i] = True

        observation = self._get_observation()
        distance_to_target = np.linalg.norm(self.robot_position - self.target_position)
        reward = -distance_to_target + collectable_reward
        terminated = distance_to_target < self.target_radius
        reward += terminated * self.config.terminal_reward

        info = {
            "distance_to_target": distance_to_target,
            "collectable_captured": sum(self.collectable_captured),
            "goal_reached": terminated,
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_observation(self) -> np.ndarray:
        return np.array(
            [
                self.robot_position[0] - self.target_position[0],
                self.robot_position[1] - self.target_position[1],
                np.cos(self.robot_angle),
                np.sin(self.robot_angle),
            ]
            + sum(
                [
                    [
                        (
                            self.robot_position[0] - self.target_position[0]
                            if self.collectable_captured[i]
                            else self.robot_position[0]
                            - self.collectable_positions[i][0]
                        ),
                        (
                            self.robot_position[1] - self.target_position[1]
                            if self.collectable_captured[i]
                            else self.robot_position[1]
                            - self.collectable_positions[i][1]
                        ),
                        self.collectable_captured[i],
                    ]
                    for i in range(len(self.collectable_captured))
                ],
                [],
            ),
            dtype=np.float32,
        )

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            raise RuntimeError(
                "Render mode must be 'human' or 'rgb_array' to draw the simulation."
            )

        self._ensure_pygame()
        pygame = self._pygame
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
        else:
            pygame.event.pump()

        width, height = self.config.window_size
        if self._surface is None:
            self._surface = pygame.Surface((width, height))

        self._surface.fill((240, 240, 240))
        self._draw_room()
        self._draw_goal()
        self._draw_collectables()
        self._draw_robot()

        if self.render_mode == "human":
            self._window.blit(self._surface, (0, 0))
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(self.metadata["render_fps"])
            return None

        frame = pygame.surfarray.array3d(self._surface)
        frame = np.transpose(frame, (1, 0, 2))
        if self._clock is not None:
            self._clock.tick(self.metadata["render_fps"])
        return frame

    def close(self) -> None:
        if self._pygame is None:
            return
        pygame = self._pygame
        if self._window is not None:
            pygame.display.quit()
            self._window = None
        if self._surface is not None:
            self._surface = None
        if self._clock is not None:
            self._clock = None
        pygame.quit()
        self._pygame = None

    def _ensure_pygame(self) -> None:
        if self.render_mode is None:
            return
        if self._pygame is not None:
            return
        import pygame  # Lazy import to keep dependency optional at runtime

        pygame.init()
        self._pygame = pygame
        if self.render_mode == "human":
            self._window = pygame.display.set_mode(self.config.window_size)
            pygame.display.set_caption("Robot Dynamics")
        else:
            self._window = None
        self._surface = pygame.Surface(self.config.window_size)
        self._clock = pygame.time.Clock()

    def _world_to_screen(self, position: np.ndarray) -> Tuple[int, int]:
        width, height = self.config.window_size
        low = float(self.config.world_low)
        high = float(self.config.world_high)
        span = max(high - low, 1e-6)
        x = int((position[0] - low) / span * width)
        y = int((1.0 - (position[1] - low) / span) * height)
        return x, y

    def _draw_room(self) -> None:
        pygame = self._pygame
        width, height = self.config.window_size
        border_color = (50, 50, 50)
        pygame.draw.rect(
            self._surface, border_color, pygame.Rect(0, 0, width, height), width=4
        )

    def _draw_goal(self) -> None:
        pygame = self._pygame
        width, height = self.config.window_size
        center = self._world_to_screen(self.target_position)
        goal_width = max(4, int(width * 0.02))
        goal_height = max(6, int(height * 0.2))
        left = int(center[0] - goal_width / 2)
        top = int(center[1] - goal_height / 2)
        left = max(0, min(left, width - goal_width))
        top = max(0, min(top, height - goal_height))
        goal_rect = pygame.Rect(left, top, goal_width, goal_height)
        pygame.draw.rect(self._surface, (50, 200, 50), goal_rect)

    def _draw_collectables(self) -> None:
        pygame = self._pygame
        for idx, position in enumerate(self.collectable_positions):
            if self.collectable_captured[idx]:
                continue
            center = self._world_to_screen(position)
            radius = self._world_to_pixels(self.config.collectable_radius)
            pygame.draw.circle(self._surface, (180, 60, 60), center, radius)

    def _draw_robot(self) -> None:
        center = self._world_to_screen(self.robot_position)
        self._draw_robot_body(center, min(*self.config.window_size))

    def _draw_robot_body(self, center: Tuple[int, int], scale: int) -> None:
        pygame = self._pygame
        body_radius = max(6, int(0.045 * scale))

        speed_ratio = float(np.clip(self._last_action[0], 0.0, 1.0))
        pulse_radius = body_radius + int(
            2 + 3 * math.sin(self._steps * 0.3 + speed_ratio * math.pi)
        )
        pulse_radius = max(pulse_radius, body_radius + 1)

        pygame.draw.circle(
            self._surface, (180, 200, 255), center, pulse_radius, width=2
        )
        pygame.draw.circle(self._surface, (60, 120, 240), center, body_radius)
        pygame.draw.circle(self._surface, (20, 60, 160), center, body_radius, width=2)
        highlight = (
            center[0] + int(body_radius * 0.4),
            center[1] - int(body_radius * 0.4),
        )
        pygame.draw.circle(
            self._surface, (220, 240, 255), highlight, max(1, body_radius // 3)
        )

        heading_world = np.array(
            [math.cos(self.robot_angle), math.sin(self.robot_angle)],
            dtype=np.float32,
        )
        heading_screen = np.array(
            [heading_world[0], -heading_world[1]], dtype=np.float32
        )
        perp_screen = np.array(
            [-heading_screen[1], heading_screen[0]], dtype=np.float32
        )

        nose_offset = heading_screen * body_radius * 1.3
        tail_offset = heading_screen * body_radius * 0.7
        wing_offset = perp_screen * body_radius * 0.8

        nose = (
            int(center[0] + nose_offset[0]),
            int(center[1] + nose_offset[1]),
        )
        left_wing = (
            int(center[0] - tail_offset[0] + wing_offset[0]),
            int(center[1] - tail_offset[1] + wing_offset[1]),
        )
        right_wing = (
            int(center[0] - tail_offset[0] - wing_offset[0]),
            int(center[1] - tail_offset[1] - wing_offset[1]),
        )

        pygame.draw.polygon(
            self._surface, (250, 250, 255), [nose, left_wing, right_wing]
        )
        pygame.draw.polygon(
            self._surface, (40, 80, 200), [nose, left_wing, right_wing], width=1
        )

        thruster_wave = (
            0.7 + 0.3 * (math.sin(self._steps * 0.5 + speed_ratio * 2.0) + 1) / 2
        )
        thruster_length = body_radius * (0.4 + 0.6 * speed_ratio) * thruster_wave
        thruster_vec = -heading_screen * thruster_length
        thruster_base = (
            int(center[0] - tail_offset[0]),
            int(center[1] - tail_offset[1]),
        )
        thruster_end = (
            int(thruster_base[0] + thruster_vec[0]),
            int(thruster_base[1] + thruster_vec[1]),
        )
        flame_color = (
            255,
            int(160 + 70 * speed_ratio),
            int(80 + 100 * speed_ratio),
        )
        pygame.draw.line(
            self._surface, flame_color, thruster_base, thruster_end, width=3
        )

    def _world_to_pixels(self, length: float) -> int:
        width, height = self.config.window_size
        low = float(self.config.world_low)
        high = float(self.config.world_high)
        span = max(high - low, 1e-6)
        return int(length / span * min(width, height))

    def _initialize_collectable_motion(self) -> None:
        count = int(self.config.n_collectables)
        if count <= 0:
            self._collectable_velocities = np.zeros((0, 2), dtype=np.float32)
            self._collectable_speed_scales = np.zeros(0, dtype=np.float32)
            return
        self._collectable_velocities = np.zeros((count, 2), dtype=np.float32)
        self._collectable_speed_scales = np.zeros(count, dtype=np.float32)
        for idx in range(count):
            speed = self._sample_collectable_speed()
            self._collectable_speed_scales[idx] = speed
            angle = float(self._rng.uniform(0.0, 2 * math.pi))
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            self._collectable_velocities[idx] = direction * speed

    def _sample_collectable_speed(self) -> float:
        if self.config.collectable_speed is not None:
            return float(self.config.collectable_speed)
        low, high = self.config.collectable_speed_range
        return float(self._rng.uniform(low, high))

    def _reflect_collectable_bounds(
        self, position: np.ndarray, velocity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        radius = float(self.config.collectable_radius)
        min_bound = self.config.world_low + radius
        max_bound = self.config.world_high - radius
        clipped = position.copy()
        for axis in range(2):
            if clipped[axis] < min_bound:
                clipped[axis] = min_bound
                velocity[axis] *= -1.0
            elif clipped[axis] > max_bound:
                clipped[axis] = max_bound
                velocity[axis] *= -1.0
        return clipped, velocity

    def _update_collectable_positions(self) -> None:
        count = int(self.config.n_collectables)
        if count <= 0:
            return
        if not hasattr(self, "collectable_captured"):
            return
        dt = self.config.control_dt
        for idx in range(count):
            if self.collectable_captured[idx]:
                if idx < len(self._collectable_velocities):
                    self._collectable_velocities[idx] = 0.0
                continue

            speed_scale = (
                self._collectable_speed_scales[idx]
                if idx < len(self._collectable_speed_scales)
                else self._sample_collectable_speed()
            )

            velocity = (
                self._collectable_velocities[idx]
                if idx < len(self._collectable_velocities)
                else np.zeros(2, dtype=np.float32)
            )

            if not np.any(velocity):
                angle = float(self._rng.uniform(0.0, 2 * math.pi))
                direction = np.array(
                    [math.cos(angle), math.sin(angle)], dtype=np.float32
                )
                velocity = direction * speed_scale

            noise_std = float(self.config.collectable_noise_std) * speed_scale
            if noise_std > 0:
                noise = self._rng.normal(0.0, noise_std, size=2).astype(np.float32)
                velocity = velocity + noise

            speed = float(np.linalg.norm(velocity))
            if speed > speed_scale and speed > 1e-6:
                velocity = velocity / speed * speed_scale
            elif speed < 1e-6:
                angle = float(self._rng.uniform(0.0, 2 * math.pi))
                direction = np.array(
                    [math.cos(angle), math.sin(angle)], dtype=np.float32
                )
                velocity = direction * speed_scale

            candidate_pos = self.collectable_positions[idx] + velocity * dt
            candidate_pos, velocity = self._reflect_collectable_bounds(
                candidate_pos, velocity
            )
            self.collectable_positions[idx] = candidate_pos
            if idx < len(self._collectable_velocities):
                self._collectable_velocities[idx] = velocity

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi


class RobotDynamicsMetricsCollector(MetricsCollector):
    def __init__(self, rolling_window_size: int = 20):
        super().__init__()
        self.rolling_window_size = rolling_window_size

    def collect_metrics_from_final_episode_info(self, info: dict, step: int):
        super().collect_metrics_from_final_episode_info(info, step)
        self.append_metric(
            "episode_stats/distance_to_target", info["distance_to_target"], step=step
        )
        self.append_metric(
            "episode_stats/collectable_captured",
            info["collectable_captured"],
            step=step,
        )
        self.append_metric(
            "episode_stats/goal_reached", float(info["goal_reached"]), step=step
        )
        self.rolling_window["collectable_captured"].append(info["collectable_captured"])
        self.rolling_window["goal_reached"].append(float(info["goal_reached"]))
        self.append_metric(
            f"episode_stats/goal_reached_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["goal_reached"]),
            step=step,
        )
        self.append_metric(
            f"episode_stats/collectable_captured_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["collectable_captured"]),
            step=step,
        )
