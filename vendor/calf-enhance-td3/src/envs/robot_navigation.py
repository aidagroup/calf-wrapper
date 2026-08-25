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
class RobotNavigationConfig:
    """Configuration values for the robot navigation environment."""

    max_steps: int = 400
    obstacle_count: int = 4
    obstacle_radius_range: Tuple[float, float] = (0.05, 0.12)
    max_speed: float = 0.15
    control_dt: float = 0.05
    max_angular_velocity: float = math.pi
    success_radius: float = 0.05
    window_size: Tuple[int, int] = (800, 600)
    obstacle_padding: float = 0.005
    obstacle_placement_attempts: int = 40
    obstacle_radius_shrink_factor: float = 0.85
    obstacle_radius_shrink_steps: int = 5
    layout_max_retries: int = 5
    obstacle_collision_penalty: float = 5.0
    moving_obstacle_count: int = 10
    moving_speed_range: Tuple[float, float] = (0.08, 0.18)
    moving_direction_change_prob: float = 0.01
    moving_obstacle_radius: Optional[float] = None
    moving_obstacle_speed: Optional[float] = None
    moving_obstacle_x_range: Optional[Tuple[float, float]] = (0.1, 0.9)
    moving_noise_std: float = 0.25
    collect_targets: bool = False
    target_count: int = 10
    target_radius: float = 0.03
    target_capture_epsilon: float = 0.08
    target_reward: float = -20.0
    target_step_penalty: float = 0.001
    heading_penalty_scale: float = 0.2


class RobotNavigationEnv(gym.Env[np.ndarray, np.ndarray]):
    """Simple 2D robot navigation environment with a top-down pygame renderer."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        config: Optional[RobotNavigationConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. Use None, 'human', or 'rgb_array'."
            )
        self.render_mode = render_mode
        self.config = config or RobotNavigationConfig()
        if self.config.control_dt <= 0.0:
            raise ValueError("RobotNavigationEnv control_dt must be positive.")

        self._rng: np.random.Generator
        self._steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)

        self.robot_position = np.zeros(2, dtype=np.float32)
        self.robot_angle = 0.0
        self.goal_position = np.array([0.0, 0.5], dtype=np.float32)
        self._obstacles = np.zeros((self.config.obstacle_count, 3), dtype=np.float32)
        self._num_obstacles = 0
        self._moving_indices: list[int] = []
        self._moving_index_order: dict[int, int] = {}
        self._moving_velocities = np.zeros((0, 2), dtype=np.float32)
        self._moving_speed_scales = np.zeros(0, dtype=np.float32)
        self._targets_remaining = 0
        self._target_total = 0
        self.trajectory: list[np.ndarray] = []
        self.captured_points: list[np.ndarray] = []
        self.show_trajectory = False
        self.show_captured_points = True
        # Screen-space light direction (points towards the light). Default: top-left.
        self.light_dir = np.array([-1.0, -1.0], dtype=np.float32)

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

        obs_dim = 2 + 2 + 2 + 3 * self.config.obstacle_count
        obs_low = np.zeros(obs_dim, dtype=np.float32)
        obs_high = np.ones_like(obs_low)
        obs_low[2:4] = -1.0
        obs_high[2:4] = 1.0
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        self._pygame = None
        self._window = None
        self._clock = None
        self._surface = None
        self._floor_surface = None
        self._floor_size = None
        self._room_geom_size = None
        self._room_geom = None

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

        # Robot starts on the right side of the room
        self.robot_position = np.array(
            [
                self._rng.uniform(0.7, 0.9),
                self._rng.uniform(0.1, 0.9),
            ],
            dtype=np.float32,
        )
        self.robot_angle = math.pi  # start roughly facing the goal

        layout_success = False
        if self.config.collect_targets:
            target_obstacles = max(1, int(self.config.target_count))
        else:
            target_obstacles = self.config.obstacle_count
        layout_retries = max(1, self.config.layout_max_retries)

        for _ in range(layout_retries):
            self._obstacles.fill(0.0)
            placed = 0
            blocker_placed = target_obstacles == 0

            if self.config.collect_targets:
                moving_count = target_obstacles
            else:
                moving_count = min(self.config.moving_obstacle_count, target_obstacles)
            if moving_count > 0:
                selection = self._rng.choice(
                    target_obstacles, size=moving_count, replace=False
                )
                selection = np.atleast_1d(selection)
                self._moving_indices = sorted(int(x) for x in selection)
                self._moving_index_order = {
                    idx: order for order, idx in enumerate(self._moving_indices)
                }
            else:
                self._moving_indices = []
                self._moving_index_order = {}
            moving_indices_set = set(self._moving_indices)

            radius_low, radius_high = self.config.obstacle_radius_range
            shrink_factor = self.config.obstacle_radius_shrink_factor
            shrink_steps = max(1, self.config.obstacle_radius_shrink_steps)
            attempts_per_obstacle = self.config.obstacle_placement_attempts

            placement_failed = False
            while placed < target_obstacles:
                success = False
                current_high = radius_high
                is_moving = placed in moving_indices_set
                force_blocker = not blocker_placed and not self.config.collect_targets

                for _ in range(shrink_steps):
                    for _ in range(attempts_per_obstacle):
                        if self.config.collect_targets:
                            radius = float(self.config.target_radius)
                        elif (
                            is_moving and self.config.moving_obstacle_radius is not None
                        ):
                            radius = float(self.config.moving_obstacle_radius)
                        else:
                            radius = float(self._rng.uniform(radius_low, current_high))

                        if force_blocker:
                            candidate = self._sample_goal_blocker_candidate(radius)
                            if candidate is None:
                                continue
                        elif is_moving:
                            order = self._moving_index_order.get(placed, 0)
                            x_low, x_high = self._moving_segment_bounds(order, radius)
                        elif self.config.collect_targets:
                            x_low = radius
                            x_high = 1.0 - radius
                        else:
                            x_low = 0.1 + radius
                            x_high = 0.45
                        if not force_blocker and x_high <= x_low:
                            x_high = x_low + 1e-3

                        if not force_blocker:
                            x = float(self._rng.uniform(x_low, x_high))
                            if self.config.collect_targets:
                                y = float(self._rng.uniform(radius, 1.0 - radius))
                            else:
                                y = float(self._rng.uniform(0.1 + radius, 0.9 - radius))
                            candidate = np.array([x, y, radius], dtype=np.float32)

                        if self._is_valid_obstacle(candidate, placed):
                            self._obstacles[placed] = candidate
                            placed += 1
                            success = True
                            if force_blocker:
                                blocker_placed = True
                            break

                    if success:
                        break
                    current_high = max(radius_low, current_high * shrink_factor)

                if not success:
                    placement_failed = True
                    break

            if placement_failed:
                continue

            self._num_obstacles = target_obstacles
            self._initialize_moving_obstacles()
            self._targets_remaining = int(
                np.count_nonzero(self._obstacles[: self._num_obstacles, 2] > 0)
            )
            self._target_total = (
                self._targets_remaining if self.config.collect_targets else 0
            )
            if not self._ensure_path_blocking_obstacle():
                placement_failed = True

            if not placement_failed:
                layout_success = True
                break

        if not layout_success:
            if self.config.collect_targets:
                self._place_targets_uniform(target_obstacles)
            else:
                raise RuntimeError(
                    "Failed to place non-overlapping obstacles; consider reducing obstacle_count or padding."
                )

        if not self.config.collect_targets:
            self._targets_remaining = 0
            self._target_total = 0

        self.trajectory = [self.robot_position.copy()]
        self.captured_points = []

        observation = self._get_observation()
        info = {
            "distance_to_goal": self._distance_to_goal(),
            "robot_angle": self.robot_angle,
        }
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._last_action = action
        self._steps += 1

        self._update_moving_obstacles()

        speed = np.clip(float(action[0]), 0.0, self.config.max_speed)
        angular_velocity = np.clip(
            float(action[1]),
            -self.config.max_angular_velocity,
            self.config.max_angular_velocity,
        )

        delta = np.array(
            [math.cos(self.robot_angle), math.sin(self.robot_angle)],
            dtype=np.float32,
        )
        self.robot_position = np.clip(
            self.robot_position + delta * (speed * self.config.control_dt),
            0.0,
            1.0,
        )
        self.robot_angle = self._wrap_angle(
            self.robot_angle + angular_velocity * self.config.control_dt
        )
        self.trajectory.append(self.robot_position.copy())

        if self.config.collect_targets:
            distance = self._distance_to_nearest_target()
        else:
            distance = self._distance_to_goal()
        distance_to_goal = self._distance_to_goal()
        heading_error = abs(self._heading_error())
        if self.config.collect_targets:
            goal_reached = distance_to_goal <= self.config.success_radius
            terminated = goal_reached
        else:
            goal_reached = distance <= self.config.success_radius
            terminated = goal_reached
        truncated = self._steps >= self.config.max_steps

        if self.config.collect_targets:
            in_obstacle = False
        else:
            in_obstacle = self._is_in_obstacle(self.robot_position)

        capture_reward = 0.0
        captured = 0
        if self.config.collect_targets:
            captured = self._handle_target_captures()
            capture_reward = captured * self.config.target_reward

        if self.config.collect_targets:
            angle_penalty = self.config.heading_penalty_scale * (
                heading_error / math.pi
            )
            reward = -distance - angle_penalty
            reward += capture_reward
            if goal_reached:
                reward += 0.0
        else:
            angle_penalty = self.config.heading_penalty_scale * (
                heading_error / math.pi
            )
            reward = -distance - angle_penalty
            if in_obstacle:
                reward -= self.config.obstacle_collision_penalty
            if goal_reached:
                reward += 1.0

        observation = self._get_observation()
        info = {
            "distance_to_goal": distance_to_goal,
            "num_obstacles": self._num_obstacles,
            "last_action": self._last_action.copy(),
            "robot_angle": self.robot_angle,
            "heading_error": heading_error,
            "in_obstacle": in_obstacle,
            "goal_reached": goal_reached,
            "moving_obstacles": len(self._moving_indices),
        }
        if self.config.collect_targets:
            info["distance_to_target"] = distance
            info["targets_remaining"] = self._targets_remaining
            info["captures"] = captured if capture_reward > 0 else 0
            info["targets_total"] = self._target_total
            info["targets_captured_total"] = max(
                0, self._target_total - self._targets_remaining
            )

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

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
                    return
        else:
            pygame.event.pump()

        width, height = self.config.window_size
        if self._surface is None:
            self._surface = pygame.Surface((width, height))

        self._draw_floor()
        self._draw_room()
        self._draw_goal()
        self._draw_obstacles()
        if self.show_trajectory:
            self._draw_trajectory()
        if self.show_captured_points:
            self._draw_captured_points()
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

    def _get_observation(self) -> np.ndarray:
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        observation[0:2] = self.robot_position
        observation[2] = math.cos(self.robot_angle)
        observation[3] = math.sin(self.robot_angle)
        observation[4:6] = self.goal_position
        obstacle_slice = 6 + 3 * self._num_obstacles
        observation[6:obstacle_slice] = self._obstacles[: self._num_obstacles].reshape(
            -1
        )
        return observation

    def _initialize_moving_obstacles(self) -> None:
        count = len(self._moving_indices)
        if count <= 0:
            self._moving_indices = []
            self._moving_index_order = {}
            self._moving_velocities = np.zeros((0, 2), dtype=np.float32)
            self._moving_speed_scales = np.zeros(0, dtype=np.float32)
            return

        self._moving_velocities = np.zeros((count, 2), dtype=np.float32)
        self._moving_speed_scales = np.zeros(count, dtype=np.float32)
        for idx in range(count):
            speed = self._sample_moving_speed()
            self._moving_speed_scales[idx] = speed
            angle = float(self._rng.uniform(0.0, 2 * math.pi))
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            self._moving_velocities[idx] = direction * speed

    def _moving_segment_bounds(self, order: int, radius: float) -> Tuple[float, float]:
        if self.config.moving_obstacle_x_range is None:
            low, high = 0.1, 0.9
        else:
            low, high = self.config.moving_obstacle_x_range

        width = max(high - low, 1e-3)
        count = max(1, len(self._moving_indices))
        segment = width / count
        start = low + segment * order
        end = min(high, start + segment)

        x_low = start + radius
        x_high = end - radius
        if x_high <= x_low:
            midpoint = (start + end) / 2.0
            x_low = max(low + radius, midpoint - 1e-3)
            x_high = min(high - radius, midpoint + 1e-3)
        return x_low, x_high

    def _reflect_on_bounds(
        self, position: np.ndarray, velocity: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        clipped = position.copy()
        for axis in range(2):
            min_bound = radius
            max_bound = 1.0 - radius
            if clipped[axis] < min_bound:
                clipped[axis] = min_bound
                velocity[axis] *= -1.0
            elif clipped[axis] > max_bound:
                clipped[axis] = max_bound
                velocity[axis] *= -1.0
        return clipped, velocity

    def _resolve_static_collisions(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: float,
        obstacle_idx: int,
        moving_set: set[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        for other_idx in range(self._num_obstacles):
            if other_idx == obstacle_idx or other_idx in moving_set:
                continue
            other = self._obstacles[other_idx]
            other_radius = other[2]
            if other_radius <= 0:
                continue
            delta = position - other[:2]
            distance = np.linalg.norm(delta)
            min_dist = radius + other_radius
            if distance < min_dist:
                if distance < 1e-6:
                    normal = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    normal = delta / distance
                penetration = min_dist - distance + 1e-4
                position = position + normal * penetration
                velocity = velocity - 2.0 * np.dot(velocity, normal) * normal
                position, velocity = self._reflect_on_bounds(position, velocity, radius)
        return position, velocity

    def _resolve_moving_collisions(self, new_positions: np.ndarray) -> None:
        count = len(self._moving_indices)
        if count < 2:
            return
        for i in range(count):
            idx_i = self._moving_indices[i]
            if idx_i >= self._num_obstacles:
                continue
            radius_i = self._obstacles[idx_i, 2]
            if radius_i <= 0:
                continue
            for j in range(i + 1, count):
                idx_j = self._moving_indices[j]
                if idx_j >= self._num_obstacles:
                    continue
                radius_j = self._obstacles[idx_j, 2]
                if radius_j <= 0:
                    continue
                delta = new_positions[idx_i] - new_positions[idx_j]
                distance = np.linalg.norm(delta)
                min_dist = radius_i + radius_j
                if distance >= min_dist:
                    continue
                if distance < 1e-6:
                    normal = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    normal = delta / max(distance, 1e-6)
                vi = self._moving_velocities[i]
                vj = self._moving_velocities[j]
                vi_n = np.dot(vi, normal)
                vj_n = np.dot(vj, normal)
                vi = vi - normal * vi_n + normal * vj_n
                vj = vj - normal * vj_n + normal * vi_n
                penetration = min_dist - distance + 1e-4
                correction = normal * (penetration / 2.0)
                new_positions[idx_i] += correction
                new_positions[idx_j] -= correction
                new_positions[idx_i], vi = self._reflect_on_bounds(
                    new_positions[idx_i], vi, radius_i
                )
                new_positions[idx_j], vj = self._reflect_on_bounds(
                    new_positions[idx_j], vj, radius_j
                )
                self._moving_velocities[i] = vi
                self._moving_velocities[j] = vj

    def _sample_goal_blocker_candidate(self, radius: float) -> Optional[np.ndarray]:
        segment = self.goal_position - self.robot_position
        seg_len = float(np.linalg.norm(segment))
        if seg_len < 1e-6:
            return None
        direction = segment / seg_len
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)

        for _ in range(64):
            t = float(self._rng.uniform(0.2, 0.8))
            base = self.robot_position + direction * (seg_len * t)
            offset = float(self._rng.uniform(-radius * 2.0, radius * 2.0))
            center = base + perpendicular * offset
            center = np.clip(center, radius, 1.0 - radius)
            candidate = np.array([center[0], center[1], radius], dtype=np.float32)
            return candidate
        return None

    def _obstacle_blocks_path(self, center: np.ndarray, radius: float) -> bool:
        start = self.robot_position
        goal = self.goal_position
        segment = goal - start
        seg_len_sq = float(np.dot(segment, segment))
        if seg_len_sq < 1e-8:
            return False
        t = float(np.dot(center - start, segment) / seg_len_sq)
        if t <= 0.0 or t >= 1.0:
            return False
        closest = start + t * segment
        distance = np.linalg.norm(center - closest)
        return distance <= radius + self.config.obstacle_padding

    def _has_path_blocking_obstacle(self) -> bool:
        for obstacle in self._obstacles[: self._num_obstacles]:
            if obstacle[2] <= 0:
                continue
            if self._obstacle_blocks_path(obstacle[:2], obstacle[2]):
                return True
        return False

    def _ensure_path_blocking_obstacle(self) -> bool:
        if (
            self.config.collect_targets
            or self._num_obstacles <= 0
            or self._has_path_blocking_obstacle()
        ):
            return True
        for idx in range(self._num_obstacles):
            radius = self._obstacles[idx, 2]
            if radius <= 0:
                continue
            candidate = self._sample_goal_blocker_candidate(radius)
            if candidate is None:
                continue
            if self._is_valid_obstacle(
                candidate, self._num_obstacles, ignore_index=idx
            ):
                self._obstacles[idx] = candidate
                return True
        return False

    def _sample_moving_speed(self) -> float:
        if self.config.moving_obstacle_speed is not None:
            return float(self.config.moving_obstacle_speed)
        speed_low, speed_high = self.config.moving_speed_range
        return float(self._rng.uniform(speed_low, speed_high))

    def _update_moving_obstacles(self) -> None:
        if not self._moving_indices:
            return
        dt = self.config.control_dt
        moving_set = set(self._moving_indices)
        new_positions = self._obstacles[:, 0:2].copy()

        for vel_idx, obstacle_idx in enumerate(self._moving_indices):
            if obstacle_idx >= self._num_obstacles:
                continue

            radius = self._obstacles[obstacle_idx, 2]
            if radius <= 0:
                continue

            speed_scale = (
                self._moving_speed_scales[vel_idx]
                if vel_idx < len(self._moving_speed_scales)
                else self._sample_moving_speed()
            )

            if vel_idx < len(self._moving_velocities):
                velocity = self._moving_velocities[vel_idx]
            else:
                velocity = np.zeros(2, dtype=np.float32)

            if not np.any(velocity):
                angle = float(self._rng.uniform(0.0, 2 * math.pi))
                direction = np.array(
                    [math.cos(angle), math.sin(angle)], dtype=np.float32
                )
                velocity = direction * speed_scale

            noise_std = float(self.config.moving_noise_std) * speed_scale
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
            position = new_positions[obstacle_idx]
            displacement = velocity * dt
            candidate_pos = position + displacement

            candidate_pos, velocity = self._reflect_on_bounds(
                candidate_pos, velocity, radius
            )
            candidate_pos, velocity = self._resolve_static_collisions(
                candidate_pos,
                velocity,
                radius,
                obstacle_idx,
                moving_set,
            )

            new_positions[obstacle_idx] = candidate_pos
            self._moving_velocities[vel_idx] = velocity

        self._resolve_moving_collisions(new_positions)

        for vel_idx, obstacle_idx in enumerate(self._moving_indices):
            if obstacle_idx >= self._num_obstacles:
                continue
            self._obstacles[obstacle_idx, 0:2] = new_positions[obstacle_idx]

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal_position - self.robot_position))

    def _nearest_target_vector(self) -> Optional[np.ndarray]:
        best_vec = None
        best_dist = float("inf")
        for obstacle in self._obstacles[: self._num_obstacles]:
            radius = obstacle[2]
            if radius <= 0:
                continue
            vec = obstacle[:2] - self.robot_position
            dist = np.linalg.norm(vec)
            if dist < best_dist:
                best_dist = dist
                best_vec = vec
        return best_vec

    def _distance_to_nearest_target(self) -> float:
        if not self.config.collect_targets:
            return self._distance_to_goal()
        vec = self._nearest_target_vector()
        if vec is None:
            return 0.0
        return float(np.linalg.norm(vec))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi) for stable orientation error calculations."""
        return float((angle + math.pi) % (2 * math.pi) - math.pi)

    def _heading_error(self) -> float:
        """Return signed angle between the robot heading and the goal direction."""
        if self.config.collect_targets:
            vec = self._nearest_target_vector()
            if vec is None or float(np.dot(vec, vec)) < 1e-12:
                return 0.0
            desired_angle = math.atan2(float(vec[1]), float(vec[0]))
        else:
            delta = self.goal_position - self.robot_position
            if float(np.dot(delta, delta)) < 1e-12:
                return 0.0
            desired_angle = math.atan2(float(delta[1]), float(delta[0]))
        return self._wrap_angle(desired_angle - self.robot_angle)

    def _is_valid_obstacle(
        self,
        candidate: np.ndarray,
        count: int,
        *,
        ignore_index: Optional[int] = None,
    ) -> bool:
        padding = self.config.obstacle_padding
        center = candidate[:2]
        radius = float(candidate[2])

        if np.any(center - radius < 0.0) or np.any(center + radius > 1.0):
            return False

        if np.linalg.norm(center - self.robot_position) < radius + padding:
            return False

        for idx, other in enumerate(self._obstacles[:count]):
            if idx == ignore_index:
                continue
            if not other.any():
                continue
            distance = np.linalg.norm(center - other[:2])
            if distance < radius + other[2] + padding:
                return False
        return True

    def _is_in_obstacle(self, position: np.ndarray) -> bool:
        for obstacle in self._obstacles[: self._num_obstacles]:
            if not obstacle.any():
                continue
            center = obstacle[:2]
            radius = obstacle[2]
            if np.linalg.norm(position - center) <= radius:
                return True
        return False

    def _handle_target_captures(self) -> int:
        if not self.config.collect_targets or self._num_obstacles <= 0:
            return 0
        captured = 0
        eps = float(self.config.target_capture_epsilon)
        for idx in range(self._num_obstacles):
            radius = self._obstacles[idx, 2]
            if radius <= 0:
                continue
            center = self._obstacles[idx, 0:2]
            threshold = max(radius, eps)
            if np.linalg.norm(self.robot_position - center) <= threshold:
                self._obstacles[idx, 2] = 0.0
                self._obstacles[idx, 0:2] = center
                captured += 1
                self.captured_points.append(center.copy())
                order = self._moving_index_order.get(idx)
                if order is not None and order < len(self._moving_velocities):
                    self._moving_velocities[order] = 0.0
        if captured > 0:
            self._targets_remaining = max(0, self._targets_remaining - captured)
        return captured

    def _place_targets_uniform(self, count: int) -> None:
        radius = float(self.config.target_radius)
        centers = self._rng.uniform(radius, 1.0 - radius, size=(count, 2))
        self._obstacles[:count, 0:2] = centers
        self._obstacles[:count, 2] = radius
        self._num_obstacles = count
        self._moving_indices = list(range(count))
        self._moving_index_order = {idx: idx for idx in range(count)}
        self._initialize_moving_obstacles()
        self._targets_remaining = count
        self._target_total = count

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
            pygame.display.set_caption("Robot Navigation")
        else:
            self._window = None
        self._surface = pygame.Surface(self.config.window_size)
        self._clock = pygame.time.Clock()

    def _world_to_screen(self, position: np.ndarray) -> Tuple[int, int]:
        floor_left, floor_top, floor_width, floor_height = self._content_rect()
        x = int(floor_left + float(position[0]) * floor_width)
        y = int(floor_top + (1.0 - float(position[1])) * floor_height)
        return x, y

    def _robot_visual_scale(self) -> float:
        return 0.425

    def _robot_visual_body_radius_px(self, scale: int) -> int:
        return max(2, int(0.045 * scale * self._robot_visual_scale()))

    def _robot_visual_clearance_px(self, scale: int) -> int:
        body_radius = self._robot_visual_body_radius_px(scale)
        size = int(body_radius * 6)
        size = max(size, int(body_radius * 4 + max(16, body_radius * 2)))
        half_diag = int(math.ceil((float(size) * math.sqrt(2.0)) / 2.0))
        return max(2, half_diag + max(2, int(body_radius * 0.4)))

    def _room_geometry(self) -> Tuple[Tuple[int, int, int, int, int, int, int, int, int], int]:
        width, height = self.config.window_size
        size = (width, height)
        if self._room_geom is not None and self._room_geom_size == size:
            return self._room_geom, int(self._room_geom[4])

        base = min(width, height)
        # Rendering-only geometry.
        #
        # `wall` controls the visible thickness (depth) of the trapezoid walls.
        # Keep the outer frame inside the window, and accept that thicker walls
        # reduce the visible floor area (but keep the mapping for robot/trajectory
        # in `_content_rect`, so visuals still look correct).
        margin = max(12, int(base * 0.04))
        wall = max(13, int(base * 0.09))

        outer_left = margin
        outer_top = margin
        outer_width = max(1, width - 2 * margin)
        outer_height = max(1, height - 2 * margin)

        min_outer = 2 * wall + max(80, int(base * 0.25))
        if outer_width < min_outer:
            outer_width = min_outer
            outer_left = max(0, (width - outer_width) // 2)
        if outer_height < min_outer:
            outer_height = min_outer
            outer_top = max(0, (height - outer_height) // 2)

        floor_left = outer_left + wall
        floor_top = outer_top + wall
        floor_width = max(1, outer_width - 2 * wall)
        floor_height = max(1, outer_height - 2 * wall)

        self._room_geom = (
            outer_left,
            outer_top,
            outer_width,
            outer_height,
            wall,
            floor_left,
            floor_top,
            floor_width,
            floor_height,
        )
        self._room_geom_size = size
        return self._room_geom, wall

    def _floor_rect(self) -> Tuple[int, int, int, int]:
        geom, _ = self._room_geometry()
        return int(geom[5]), int(geom[6]), int(geom[7]), int(geom[8])

    def _draw_outer_frame(self, outer: "pygame.Rect", wall: int) -> None:
        pygame = self._pygame
        width, height = self.config.window_size

        # A thick "bezel" outside the room makes the wall edge read as solid material,
        # not a thin cardboard line.
        bezel = max(14, int(wall * 0.55))
        frame_outer = outer.inflate(bezel * 2, bezel * 2)

        fx = pygame.Surface((width, height), flags=pygame.SRCALPHA)
        # Outside the room should read as the same ground plane as the floor.
        floor_h = max(1, outer.height - 2 * wall)
        base = max(1, min(outer.width - 2 * wall, floor_h))
        floor_left = outer.left + wall
        floor_top = outer.top + wall

        floor_base = (220, 222, 228)
        pygame.draw.rect(fx, (*floor_base, 255), frame_outer)

        minor_step = max(18, int(base * 0.045))
        major_step = minor_step * 4
        minor = (212, 214, 222, 255)
        major = (195, 198, 208, 255)

        x = floor_left
        while x >= frame_outer.left:
            x -= minor_step
        while x <= frame_outer.right:
            pygame.draw.line(fx, minor, (x, frame_outer.top), (x, frame_outer.bottom), width=1)
            x += minor_step

        y = floor_top
        while y >= frame_outer.top:
            y -= minor_step
        while y <= frame_outer.bottom:
            pygame.draw.line(fx, minor, (frame_outer.left, y), (frame_outer.right, y), width=1)
            y += minor_step

        x = floor_left
        while x >= frame_outer.left:
            x -= major_step
        while x <= frame_outer.right:
            pygame.draw.line(fx, major, (x, frame_outer.top), (x, frame_outer.bottom), width=2)
            x += major_step

        y = floor_top
        while y >= frame_outer.top:
            y -= major_step
        while y <= frame_outer.bottom:
            pygame.draw.line(fx, major, (frame_outer.left, y), (frame_outer.right, y), width=2)
            y += major_step

        pygame.draw.rect(fx, (0, 0, 0, 0), outer, border_radius=0)

        self._surface.blit(fx, (0, 0))

        # Inner cap band on the wall itself.
        outer_rim_w = max(6, int(wall * 0.19))
        pygame.draw.rect(
            self._surface, (250, 250, 252), outer, width=outer_rim_w, border_radius=0
        )

    def _content_rect(self) -> Tuple[int, int, int, int]:
        floor_left, floor_top, floor_w, floor_h = self._floor_rect()
        base = min(floor_w, floor_h)
        if base <= 1:
            return floor_left, floor_top, floor_w, floor_h

        max_margin = max(0, base // 2 - 2)
        margin_factor = 0.55
        margin = min(int(self._robot_visual_clearance_px(base) * margin_factor), max_margin)
        for _ in range(2):
            content_w = max(1, floor_w - 2 * margin)
            content_h = max(1, floor_h - 2 * margin)
            scale = min(content_w, content_h)
            next_margin = min(
                int(self._robot_visual_clearance_px(scale) * margin_factor), max_margin
            )
            if abs(next_margin - margin) <= 1:
                break
            margin = next_margin

        return (
            floor_left + margin,
            floor_top + margin,
            max(1, floor_w - 2 * margin),
            max(1, floor_h - 2 * margin),
        )

    def _light_dir_unit(self) -> Tuple[float, float]:
        ld = np.asarray(self.light_dir, dtype=np.float32).reshape(-1)
        if ld.size < 2:
            return (-0.7071, -0.7071)
        x = float(ld[0])
        y = float(ld[1])
        norm = math.hypot(x, y)
        if norm < 1e-6:
            return (-0.7071, -0.7071)
        return (x / norm, y / norm)

    def _shadow_offset_px(self, scale: int) -> Tuple[int, int]:
        lx, ly = self._light_dir_unit()
        sx, sy = (-lx, -ly)
        length = max(4, int(scale * 0.018))
        return (int(sx * length), int(sy * length))

    def _draw_floor(self) -> None:
        pygame = self._pygame
        width, height = self.config.window_size
        self._surface.fill((190, 192, 198))

        floor_left, floor_top, floor_w, floor_h = self._floor_rect()
        size = (floor_w, floor_h)
        if self._floor_surface is None or self._floor_size != size:
            floor = pygame.Surface(size)

            floor.fill((220, 222, 228))

            # Grid (Unity-like: major + minor)
            minor = (212, 214, 222)
            major = (195, 198, 208)
            base = min(floor_w, floor_h)
            minor_step = max(18, int(base * 0.045))
            major_step = minor_step * 4
            for x in range(0, floor_w + 1, minor_step):
                pygame.draw.line(floor, minor, (x, 0), (x, floor_h), width=1)
            for y in range(0, floor_h + 1, minor_step):
                pygame.draw.line(floor, minor, (0, y), (floor_w, y), width=1)
            for x in range(0, floor_w + 1, major_step):
                pygame.draw.line(floor, major, (x, 0), (x, floor_h), width=2)
            for y in range(0, floor_h + 1, major_step):
                pygame.draw.line(floor, major, (0, y), (floor_w, y), width=2)

            self._floor_surface = floor
            self._floor_size = size

        self._surface.blit(self._floor_surface, (floor_left, floor_top))

    def _draw_room(self) -> None:
        pygame = self._pygame
        width, height = self.config.window_size
        geom, wall = self._room_geometry()
        outer = pygame.Rect(int(geom[0]), int(geom[1]), int(geom[2]), int(geom[3]))
        floor = pygame.Rect(int(geom[5]), int(geom[6]), int(geom[7]), int(geom[8]))

        shadow = pygame.Surface((width, height), flags=pygame.SRCALPHA)
        sx, sy = self._shadow_offset_px(min(floor.width, floor.height))
        shadow_rect = outer.move(int(sx * 0.8), int(sy * 0.9))
        pygame.draw.rect(shadow, (0, 0, 0, 70), shadow_rect, border_radius=6)
        self._surface.blit(shadow, (0, 0))

        def lerp_point(a: Tuple[int, int], b: Tuple[int, int], t: float) -> Tuple[int, int]:
            return (int(a[0] * (1.0 - t) + b[0] * t), int(a[1] * (1.0 - t) + b[1] * t))

        def lerp_color(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
            return (
                int(a[0] * (1.0 - t) + b[0] * t),
                int(a[1] * (1.0 - t) + b[1] * t),
                int(a[2] * (1.0 - t) + b[2] * t),
            )

        def shaded_trapezoid(
            inner_a: Tuple[int, int],
            inner_b: Tuple[int, int],
            outer_a: Tuple[int, int],
            outer_b: Tuple[int, int],
            inner_color: Tuple[int, int, int],
            outer_color: Tuple[int, int, int],
        ) -> None:
            steps = max(10, int(wall * 0.35))
            for i in range(steps):
                t0 = i / steps
                t1 = (i + 1) / steps
                a0 = lerp_point(inner_a, outer_a, t0)
                b0 = lerp_point(inner_b, outer_b, t0)
                a1 = lerp_point(inner_a, outer_a, t1)
                b1 = lerp_point(inner_b, outer_b, t1)
                col = lerp_color(inner_color, outer_color, (t0 + t1) * 0.5)
                pygame.draw.polygon(self._surface, col, [a0, b0, b1, a1])

        # Wall faces as trapezoids between outer and floor rects.
        top_inner_a = (floor.left, floor.top)
        top_inner_b = (floor.right, floor.top)
        top_outer_a = (outer.left, outer.top)
        top_outer_b = (outer.right, outer.top)
        shaded_trapezoid(top_inner_a, top_inner_b, top_outer_a, top_outer_b, (95, 98, 112), (132, 136, 150))

        left_inner_a = (floor.left, floor.bottom)
        left_inner_b = (floor.left, floor.top)
        left_outer_a = (outer.left, outer.bottom)
        left_outer_b = (outer.left, outer.top)
        shaded_trapezoid(left_inner_a, left_inner_b, left_outer_a, left_outer_b, (82, 85, 98), (118, 122, 136))

        right_inner_a = (floor.right, floor.top)
        right_inner_b = (floor.right, floor.bottom)
        right_outer_a = (outer.right, outer.top)
        right_outer_b = (outer.right, outer.bottom)
        shaded_trapezoid(right_inner_a, right_inner_b, right_outer_a, right_outer_b, (62, 64, 74), (98, 102, 116))

        bottom_inner_a = (floor.right, floor.bottom)
        bottom_inner_b = (floor.left, floor.bottom)
        bottom_outer_a = (outer.right, outer.bottom)
        bottom_outer_b = (outer.left, outer.bottom)
        shaded_trapezoid(bottom_inner_a, bottom_inner_b, bottom_outer_a, bottom_outer_b, (52, 54, 62), (82, 86, 98))

        # Inner seam + contact shadow (also scale with wall thickness).
        seam = pygame.Surface((width, height), flags=pygame.SRCALPHA)
        # Keep it subtle: this reads as a "baseboard/plinth" at the wall-floor junction.
        seam_inflate = max(3, int(wall * 0.07))
        seam_w = max(4, int(wall * 0.12))
        pygame.draw.rect(
            seam,
            (0, 0, 0, 45),
            floor.inflate(seam_inflate, seam_inflate),
            width=seam_w,
            border_radius=4,
        )
        self._surface.blit(seam, (0, 0))
        inner_outline_w = max(2, int(wall * 0.05))
        pygame.draw.rect(
            self._surface, (20, 22, 28), floor, width=inner_outline_w, border_radius=4
        )

        self._draw_outer_frame(outer, wall)

    def _draw_goal(self) -> None:
        pygame = self._pygame
        geom, wall = self._room_geometry()
        outer = pygame.Rect(int(geom[0]), int(geom[1]), int(geom[2]), int(geom[3]))
        floor_left, floor_top, floor_w, floor_h = self._floor_rect()
        floor = pygame.Rect(int(floor_left), int(floor_top), int(floor_w), int(floor_h))
        content_left, content_top, content_w, content_h = self._content_rect()
        content = pygame.Rect(int(content_left), int(content_top), int(content_w), int(content_h))
        scale = min(content.width, content.height)

        goal_px = self._world_to_screen(self.goal_position)
        notch_h = max(int(scale * 0.065), int(wall * 0.75))
        half_h = max(6, notch_h // 2)
        center_y = int(np.clip(goal_px[1], floor.top + half_h + 2, floor.bottom - half_h - 2))
        y_top_floor = center_y - half_h
        y_bot_floor = center_y + half_h

        def map_floor_y_to_outer_y(floor_y: int) -> int:
            t = (floor_y - floor.top) / max(1.0, float(floor.height))
            return int(outer.top + t * float(outer.height))

        y_top_outer = map_floor_y_to_outer_y(y_top_floor)
        y_bot_outer = map_floor_y_to_outer_y(y_bot_floor)

        wall_depth = max(1, floor.left - outer.left)
        notch_shift_left_px = max(3, int(wall_depth * 0.08))
        notch_nudge_right_px = 2
        x_outer = floor.left - max(2, int(wall_depth * 0.22)) - notch_shift_left_px + notch_nudge_right_px - 4
        x_inner = floor.left + max(1, int(wall_depth * 0.04)) - notch_shift_left_px + notch_nudge_right_px
        taper = max(1, int(half_h * 0.18))
        outer_base_extra_px = max(2, int(half_h * 0.12))
        right_base_extra_px = max(2, int(half_h * 0.10))
        notch_poly = [
            (x_outer, y_top_outer - outer_base_extra_px),
            (x_outer, y_bot_outer + outer_base_extra_px),
            (x_inner, y_bot_floor - taper + right_base_extra_px),
            (x_inner, y_top_floor + taper - right_base_extra_px),
        ]
        pygame.draw.polygon(self._surface, (0, 0, 0), notch_poly)

    def _draw_obstacles(self) -> None:
        pygame = self._pygame
        floor_left, floor_top, floor_w, floor_h = self._content_rect()
        scale = min(floor_w, floor_h)
        sx, sy = self._shadow_offset_px(scale)
        lx, ly = self._light_dir_unit()
        for idx in range(self._num_obstacles):
            x, y, radius = self._obstacles[idx]
            center = self._world_to_screen(np.array([x, y], dtype=np.float32))
            px_radius = int(radius * scale)
            shadow_center = (center[0] + sx, center[1] + sy)
            pygame.draw.circle(
                self._surface,
                (120, 120, 130),
                shadow_center,
                max(1, px_radius),
            )
            pygame.draw.circle(self._surface, (235, 140, 45), center, max(1, px_radius))
            pygame.draw.circle(
                self._surface, (150, 82, 20), center, max(1, px_radius), width=2
            )
            highlight = (
                int(center[0] + lx * px_radius * 0.35),
                int(center[1] + ly * px_radius * 0.35),
            )
            pygame.draw.circle(
                self._surface, (255, 220, 165), highlight, max(1, px_radius // 3)
            )

    def _draw_robot(self) -> None:
        floor_left, floor_top, floor_w, floor_h = self._content_rect()
        scale = min(floor_w, floor_h)
        self._draw_robot_body(self._world_to_screen(self.robot_position), scale)

    def _draw_trajectory(self) -> None:
        if len(self.trajectory) < 2:
            return
        pygame = self._pygame
        floor_left, floor_top, floor_w, floor_h = self._content_rect()
        scale = min(floor_w, floor_h)
        points = [self._world_to_screen(p) for p in self.trajectory]
        color = (20, 40, 80)
        line_width = max(2, int(scale * 0.004))
        dash = max(10, int(scale * 0.02))
        gap = max(7, int(scale * 0.014))

        for start, end in zip(points[:-1], points[1:]):
            x1, y1 = start
            x2, y2 = end
            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            step = dash + gap
            ux = dx / dist
            uy = dy / dist
            t = 0.0
            while t < dist:
                t_end = min(dist, t + dash)
                sx = int(x1 + ux * t)
                sy = int(y1 + uy * t)
                ex = int(x1 + ux * t_end)
                ey = int(y1 + uy * t_end)
                pygame.draw.line(
                    self._surface, color, (sx, sy), (ex, ey), width=line_width
                )
                t += step

    def _draw_captured_points(self) -> None:
        if not self.captured_points:
            return
        pygame = self._pygame
        floor_left, floor_top, floor_w, floor_h = self._content_rect()
        scale = min(floor_w, floor_h)
        radius = max(7, int(scale * 0.018))
        stroke = max(2, radius // 4)
        tick = radius + max(4, radius // 2)
        for point in self.captured_points:
            x, y = self._world_to_screen(point)
            pygame.draw.circle(self._surface, (255, 220, 40), (x, y), radius)
            pygame.draw.circle(
                self._surface, (20, 20, 20), (x, y), radius, width=stroke
            )
            pygame.draw.line(
                self._surface,
                (20, 20, 20),
                (x - tick, y),
                (x + tick, y),
                width=stroke,
            )
            pygame.draw.line(
                self._surface,
                (20, 20, 20),
                (x, y - tick),
                (x, y + tick),
                width=stroke,
            )

    def _draw_robot_body(self, center: Tuple[int, int], scale: int) -> None:
        pygame = self._pygame
        body_radius = self._robot_visual_body_radius_px(scale)

        speed_ratio = float(
            np.clip(abs(float(np.asarray(self._last_action).reshape(-1)[0])), 0.0, 1.0)
        )

        # Shadow (screen-space, consistent with light direction; does not rotate with the robot).
        sx, sy = self._shadow_offset_px(scale)
        shadow_w = int(body_radius * 4.6)
        shadow_h = int(body_radius * 3.6)
        shadow_rect = pygame.Rect(0, 0, shadow_w, shadow_h)
        shadow_rect.center = (center[0] + sx, center[1] + sy)
        shadow_layer = pygame.Surface((shadow_w, shadow_h), flags=pygame.SRCALPHA)
        shadow_layer.fill((0, 0, 0, 0))
        pygame.draw.ellipse(
            shadow_layer, (0, 0, 0, 60), pygame.Rect(0, 0, shadow_w, shadow_h)
        )
        self._surface.blit(shadow_layer, shadow_rect.topleft)

        # Build a small top-down sprite then rotate it.
        size = int(body_radius * 6)
        size = max(size, int(body_radius * 4 + max(16, body_radius * 2)))
        sprite = pygame.Surface((size, size), flags=pygame.SRCALPHA)
        cx = cy = size // 2

        # Side tracks (top-down). Rectangles read better than wheels from above.
        track_len = int(body_radius * 4.2)
        track_thickness = int(body_radius * 1.35)
        track_offset = int(body_radius * 2.25)
        track_color = (35, 35, 40)
        track_inner = (80, 80, 90)
        track_outline = (15, 15, 18)
        accent = (250, 140, 40)
        outline_w = max(1, int(body_radius * 0.18))
        inner_w = max(1, int(body_radius * 0.22))

        for side in (-1, 1):
            track_rect = pygame.Rect(0, 0, track_len, track_thickness)
            track_rect.center = (cx, cy + side * track_offset)
            radius = max(2, track_rect.height // 2)
            pygame.draw.rect(sprite, track_color, track_rect, border_radius=radius)
            pygame.draw.rect(
                sprite, track_outline, track_rect, width=outline_w, border_radius=radius
            )

            shrink_x = max(6, int(track_rect.width * 0.12))
            shrink_y = max(4, int(track_rect.height * 0.35))
            inner_rect = track_rect.inflate(-shrink_x, -shrink_y)
            if inner_rect.width > 2 and inner_rect.height > 2:
                inner_radius = max(2, inner_rect.height // 2)
                pygame.draw.rect(
                    sprite, track_inner, inner_rect, width=inner_w, border_radius=inner_radius
                )
                accent_y = inner_rect.centery
                pygame.draw.line(
                    sprite,
                    accent,
                    (inner_rect.left + int(inner_rect.width * 0.18), accent_y),
                    (inner_rect.right - int(inner_rect.width * 0.18), accent_y),
                    width=max(1, int(track_rect.height * 0.12)),
                )

        # Main body shell
        body_w = int(body_radius * 3.6)
        body_h = int(body_radius * 3.0)
        body_rect = pygame.Rect(0, 0, body_w, body_h)
        body_rect.center = (cx, cy)
        pygame.draw.ellipse(sprite, (248, 248, 252), body_rect)
        pygame.draw.ellipse(
            sprite, (145, 150, 162), body_rect, width=max(1, int(body_radius * 0.12))
        )

        # Orange accent lines
        stripe_w = max(2, body_radius // 5)
        pygame.draw.line(
            sprite,
            accent,
            (body_rect.left + int(body_w * 0.22), body_rect.top + int(body_h * 0.22)),
            (body_rect.left + int(body_w * 0.22), body_rect.bottom - int(body_h * 0.22)),
            width=stripe_w,
        )
        pygame.draw.line(
            sprite,
            accent,
            (body_rect.right - int(body_w * 0.22), body_rect.top + int(body_h * 0.22)),
            (body_rect.right - int(body_w * 0.22), body_rect.bottom - int(body_h * 0.22)),
            width=stripe_w,
        )

        # Visor panel (front is +X in sprite space)
        visor_w = int(body_radius * 2.5)
        visor_h = int(body_radius * 1.6)
        visor_rect = pygame.Rect(0, 0, visor_w, visor_h)
        visor_rect.center = (cx + int(body_radius * 0.55), cy)
        pygame.draw.ellipse(sprite, (18, 20, 26), visor_rect)
        pygame.draw.ellipse(
            sprite, (90, 95, 110), visor_rect, width=max(1, int(body_radius * 0.12))
        )

        # Eyes glow + small smile
        glow = (90, 235, 255)
        eye_r = max(2, int(body_radius * 0.28))
        eye_dx = int(body_radius * 0.35)
        eye_dy = int(body_radius * 0.20)
        left_eye = (visor_rect.centerx - eye_dx, visor_rect.centery - eye_dy)
        right_eye = (visor_rect.centerx + eye_dx, visor_rect.centery - eye_dy)
        for ex, ey in (left_eye, right_eye):
            pygame.draw.circle(sprite, (*glow, 70), (ex, ey), int(eye_r * 2.2))
            pygame.draw.circle(sprite, (*glow, 140), (ex, ey), int(eye_r * 1.6))
            pygame.draw.circle(sprite, glow, (ex, ey), eye_r)
        smile_rect = pygame.Rect(0, 0, int(body_radius * 0.9), int(body_radius * 0.55))
        smile_rect.center = (visor_rect.centerx, visor_rect.centery + int(body_radius * 0.30))
        pygame.draw.arc(
            sprite,
            glow,
            smile_rect,
            math.pi * 0.10,
            math.pi * 0.90,
            width=max(1, int(body_radius * 0.12)),
        )

        # Chest glow core
        core_center = (cx - int(body_radius * 0.35), cy)
        core_r = max(2, int(body_radius * 0.42))
        pygame.draw.circle(sprite, (*glow, 55), core_center, int(core_r * 2.0))
        pygame.draw.circle(sprite, (*glow, 120), core_center, int(core_r * 1.4))
        pygame.draw.circle(sprite, glow, core_center, core_r)
        pygame.draw.circle(
            sprite,
            accent,
            core_center,
            max(2, core_r + max(2, int(body_radius * 0.35))),
            width=max(1, int(body_radius * 0.12)),
        )

        # Aura pulse around robot (screen-space after rotate)
        pulse = 2 + 3 * math.sin(self._steps * 0.3 + speed_ratio * math.pi)
        pulse_radius = int(body_radius * 2.0 + pulse)
        pygame.draw.circle(
            sprite,
            (150, 200, 255, 95),
            (cx, cy),
            pulse_radius,
            width=max(1, int(body_radius * 0.12)),
        )

        # Thruster flame behind body (back is -X in sprite space)
        wave = 0.7 + 0.3 * (math.sin(self._steps * 0.5 + speed_ratio * 2.0) + 1) / 2
        flame_len = int(body_radius * (0.7 + 1.0 * speed_ratio) * wave)
        flame_base = (body_rect.left + int(body_radius * 0.15), cy)
        flame_end = (flame_base[0] - flame_len, flame_base[1])
        flame_color = (255, int(160 + 70 * speed_ratio), int(80 + 100 * speed_ratio), 210)
        flame_w = max(1, int(body_radius * 0.25))
        pygame.draw.line(sprite, flame_color, flame_base, flame_end, width=flame_w)
        pygame.draw.circle(sprite, flame_color, flame_end, max(1, flame_w))

        angle_deg = float(self.robot_angle * 180.0 / math.pi)
        rotated = pygame.transform.rotozoom(sprite, angle_deg, 1.0)
        rect = rotated.get_rect(center=center)
        self._surface.blit(rotated, rect)


class SimpleGoalController:
    """Greedy controller that moves toward the goal with speed/angular-velocity actions."""

    def __init__(
        self,
        *,
        max_speed: float,
        turn_gain: float = 1.0,
        max_turn_rate: float = math.pi,
    ) -> None:
        self.max_speed = float(max_speed)
        self.turn_gain = float(turn_gain)
        self.max_turn_rate = float(max_turn_rate)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + math.pi) % (2 * math.pi) - math.pi)

    def act(self, observation: np.ndarray) -> np.ndarray:
        robot = observation[0:2]
        heading_cos = float(observation[2])
        heading_sin = float(observation[3])
        current_angle = math.atan2(heading_sin, heading_cos)
        goal = observation[4:6]
        delta = goal - robot

        distance = np.linalg.norm(delta)
        if distance < 1e-8:
            return np.array([0.0, 0.0], dtype=np.float32)

        desired_angle = float(math.atan2(delta[1], delta[0]))
        angle_error = self._wrap_angle(desired_angle - current_angle)
        angular_velocity = np.clip(
            self.turn_gain * angle_error,
            -self.max_turn_rate,
            self.max_turn_rate,
        )

        # Saturate speed so the robot slows as it nears the goal
        if self.max_speed <= 0.0:
            speed_ratio = 0.0
        else:
            speed_ratio = min(1.0, distance / (self.max_speed * 8.0))
        speed_ratio = float(np.clip(speed_ratio, 0.0, 1.0))
        return np.array([speed_ratio, angular_velocity], dtype=np.float32)


class RobotNavigationMetricsCollector(MetricsCollector):
    def __init__(self, rolling_window_size: int = 20):
        super().__init__()
        self.rolling_window_size = rolling_window_size

    def collect_metrics_from_final_episode_info(self, info: dict, step: int) -> dict:
        super().collect_metrics_from_final_episode_info(info, step)
        self.append_metric(
            "episode_stats/distance_to_goal",
            info.get("distance_to_goal", 0.0),
            step=step,
        )
        self.rolling_window["distance_to_goal"].append(
            info.get("distance_to_goal", 0.0)
        )
        self.append_metric(
            f"episode_stats/distance_to_goal_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["distance_to_goal"]),
            step=step,
        )
        self.append_metric(
            "episode_stats/in_obstacle",
            float(info.get("in_obstacle", False)),
            step=step,
        )
        self.rolling_window["in_obstacle"].append(float(info.get("in_obstacle", False)))
        self.append_metric(
            f"episode_stats/in_obstacle_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["in_obstacle"]),
            step=step,
        )
        if "targets_captured_total" in info:
            captures = float(info.get("targets_captured_total", 0.0))
            self.append_metric("episode_stats/targets_captured", captures, step=step)
            self.rolling_window["targets_captured"].append(captures)
            self.append_metric(
                f"episode_stats/targets_captured_rolling_{self.rolling_window_size}",
                np.mean(self.rolling_window["targets_captured"]),
                step=step,
            )
        self.append_metric(
            "episode_stats/goal_reached",
            float(info.get("goal_reached", False)),
            step=step,
        )
        self.rolling_window["goal_reached"].append(
            float(info.get("goal_reached", False))
        )
        self.append_metric(
            f"episode_stats/goal_reached_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["goal_reached"]),
            step=step,
        )
