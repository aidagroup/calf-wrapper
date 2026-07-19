import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class LidarNavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, seed=None):
        super().__init__()

        self.width = 10.0
        self.height = 5.0
        self.goal_pos = np.array([1.0, 2.5])
        self.goal_radius = 0.3
        self.trajectory = []  # Store robot positions

        self.robot_radius = 0.2
        self.lidar_range = 5.0
        self.num_lidar_beams = 16

        self.max_velocity = 1.0
        self.max_angular_velocity = np.pi

        self.observation_space = spaces.Box(
            low=0,
            high=self.lidar_range,
            shape=(self.num_lidar_beams + 3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_angular_velocity]),
            high=np.array([self.max_velocity, self.max_angular_velocity]),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 600
        self.robot_pos = np.array(
            [np.random.uniform(5.5, 9.5), np.random.uniform(0.5, 4.5)]
        )
        self._create_static_map()
        self.reset(seed=seed)

    def _create_static_map(self):
        # Generate random obstacles in the left half of the map
        self.obstacles = []

        tau = np.linspace(0, 1, 100)[:, None]
        segment = self.robot_pos[None, :] * tau + (1 - tau) * self.goal_pos[None, :]
        segment = segment[
            (segment[:, 0] < 4.5) & (segment[:, 0] > self.goal_pos[0] + 0.5)
        ]
        idx = np.random.randint(0, len(segment) - 1)
        x, y = segment[idx]
        w, h = np.random.uniform(0.5, 1.0), np.random.uniform(0.5, 1.0)
        self.obstacles.append(((x, y - h / 2), (x + w, y + h / 2)))

        for _ in range(4):
            collision = True
            attempts = 0
            while collision and attempts < 50:
                w, h = np.random.uniform(0.5, 1.0), np.random.uniform(0.5, 1.0)
                x = np.random.uniform(1.3, 4.5 - w)
                y = np.random.uniform(0.5, self.height - 0.5 - h)
                new_obstacle = ((x, y), (x + w, y + h))

                # Check collision with existing obstacles
                collision = False
                for obs in self.obstacles:
                    if self._rectangles_overlap(new_obstacle, obs):
                        collision = True
                        break

                if not collision:
                    self.obstacles.append(new_obstacle)
                attempts += 1

    def _rectangles_overlap(self, rect1, rect2):
        (x1_min, y1_min), (x1_max, y1_max) = rect1
        (x2_min, y2_min), (x2_max, y2_max) = rect2

        # Check if one rectangle is to the left of the other
        if x1_max < x2_min or x2_max < x1_min:
            return False

        # Check if one rectangle is above the other
        if y1_max < y2_min or y2_max < y1_min:
            return False

        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array(
            [np.random.uniform(5.5, 9.5), np.random.uniform(0.5, 4.5)]
        )
        self._create_static_map()
        self.robot_angle = np.random.uniform(0, 2 * np.pi)
        self.step_count = 0
        self.trajectory = [self.robot_pos.copy()]
        self.colliding_obstacle = None
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        v, omega = action
        dt = 0.05

        # Apply motion
        self.robot_angle += omega * dt
        self.robot_pos[0] += v * np.cos(self.robot_angle) * dt
        self.robot_pos[1] += v * np.sin(self.robot_angle) * dt
        self.robot_angle %= 2 * np.pi

        # Clip to room
        self.robot_pos[0] = np.clip(self.robot_pos[0], 0, self.width)
        self.robot_pos[1] = np.clip(self.robot_pos[1], 0, self.height)

        # Store position for trajectory
        self.trajectory.append(self.robot_pos.copy())

        # Reward and termination

        done = False

        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        reward = -0.1 * dist_to_goal
        if dist_to_goal < self.goal_radius:
            reward += 10.0
            done = True

        if self._check_collision():
            reward -= 15.0  # Strong penalty but does not end episode

        self.step_count += 1
        if self.step_count > 500:
            done = True

        return self._get_obs(), reward, done, False, {}

    def _check_collision(self):
        x, y = self.robot_pos
        self.colliding_obstacle = None
        for (x0, y0), (x1, y1) in self.obstacles:
            if x0 < x < x1 and y0 < y < y1:
                self.colliding_obstacle = ((x0, y0), (x1, y1))
                return True
        return False

    def _simulate_lidar(self):
        readings = []
        hit_points = []
        for i in range(self.num_lidar_beams):
            angle = self.robot_angle + i * 2 * np.pi / self.num_lidar_beams
            min_dist = self.lidar_range
            hit_point = self.robot_pos + min_dist * np.array(
                [np.cos(angle), np.sin(angle)]
            )

            # Check wall collisions first
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            # Horizontal walls (y = 0 and y = height)
            if ray_dir[1] != 0:
                t1 = -self.robot_pos[1] / ray_dir[1]  # Time to hit y = 0
                t2 = (self.height - self.robot_pos[1]) / ray_dir[
                    1
                ]  # Time to hit y = height
                for t in [t1, t2]:
                    if t > 0:
                        x = self.robot_pos[0] + t * ray_dir[0]
                        if 0 <= x <= self.width and t < min_dist:
                            min_dist = t
                            hit_point = self.robot_pos + t * ray_dir

            # Vertical walls (x = 0 and x = width)
            if ray_dir[0] != 0:
                t1 = -self.robot_pos[0] / ray_dir[0]  # Time to hit x = 0
                t2 = (self.width - self.robot_pos[0]) / ray_dir[
                    0
                ]  # Time to hit x = width
                for t in [t1, t2]:
                    if t > 0:
                        y = self.robot_pos[1] + t * ray_dir[1]
                        if 0 <= y <= self.height and t < min_dist:
                            min_dist = t
                            hit_point = self.robot_pos + t * ray_dir

            # Check obstacle collisions
            for (x0, y0), (x1, y1) in self.obstacles:
                # Skip the obstacle we're inside of
                if (
                    self.colliding_obstacle is not None
                    and (x0, y0) == self.colliding_obstacle[0]
                    and (x1, y1) == self.colliding_obstacle[1]
                ):
                    continue

                # Check intersection with vertical edges of obstacle
                if ray_dir[0] != 0:
                    for x in [x0, x1]:
                        t = (x - self.robot_pos[0]) / ray_dir[0]
                        if t > 0:
                            y = self.robot_pos[1] + t * ray_dir[1]
                            if y0 <= y <= y1 and t < min_dist:
                                min_dist = t
                                hit_point = self.robot_pos + t * ray_dir

                # Check intersection with horizontal edges of obstacle
                if ray_dir[1] != 0:
                    for y in [y0, y1]:
                        t = (y - self.robot_pos[1]) / ray_dir[1]
                        if t > 0:
                            x = self.robot_pos[0] + t * ray_dir[0]
                            if x0 <= x <= x1 and t < min_dist:
                                min_dist = t
                                hit_point = self.robot_pos + t * ray_dir

            readings.append(min_dist)
            hit_points.append(hit_point)
        return np.array(readings), np.array(hit_points)

    def _get_obs(self):
        lidar_readings, _ = self._simulate_lidar()
        to_goal = self.goal_pos - self.robot_pos
        distance = np.linalg.norm(to_goal)
        angle = np.arctan2(to_goal[1], to_goal[0]) - self.robot_angle
        return np.concatenate(
            [lidar_readings, [distance, np.sin(angle), np.cos(angle)]], dtype=np.float32
        )

    def _is_point_in_any_obstacle(self, point):
        x, y = point
        for (x0, y0), (x1, y1) in self.obstacles:
            if x0 < x < x1 and y0 < y < y1:
                return True
        return False

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.window_size, int(self.window_size * self.height / self.width))
            )
            self.clock = pygame.time.Clock()

        if self.render_mode is None:
            return

        surface = pygame.Surface(
            (self.window_size, int(self.window_size * self.height / self.width))
        )
        surface.fill((255, 255, 240))

        def world_to_px(pos):
            scale = self.window_size / self.width
            return int(pos[0] * scale), int(pos[1] * scale)

        # Draw obstacles
        for (x0, y0), (x1, y1) in self.obstacles:
            rect = pygame.Rect(
                *world_to_px((x0, y0)),
                (x1 - x0) * self.window_size / self.width,
                (y1 - y0) * self.window_size / self.width,
            )
            pygame.draw.rect(surface, (0, 0, 0), rect)

        # Draw goal
        goal_px = world_to_px(self.goal_pos)
        pygame.draw.circle(
            surface,
            (0, 200, 0),
            goal_px,
            int(self.goal_radius * self.window_size / self.width),
        )

        # Draw trajectory with color changes
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                start_pos = self.trajectory[i]
                end_pos = self.trajectory[i + 1]
                start_px = world_to_px(start_pos)
                end_px = world_to_px(end_pos)

                # Choose color based on whether points are in obstacle
                start_in_obstacle = self._is_point_in_any_obstacle(start_pos)
                end_in_obstacle = self._is_point_in_any_obstacle(end_pos)

                if start_in_obstacle or end_in_obstacle:
                    color = (255, 0, 0)  # Red for segments in obstacle
                else:
                    color = (100, 100, 255)  # Original blue color

                pygame.draw.line(surface, color, start_px, end_px, 2)

        # Draw lidar rays
        _, hit_points = self._simulate_lidar()
        for hit_point in hit_points:
            pygame.draw.line(
                surface,
                (255, 165, 0),
                world_to_px(self.robot_pos),
                world_to_px(hit_point),
                1,
            )

        # Draw robot
        heading = np.array([np.cos(self.robot_angle), np.sin(self.robot_angle)])
        left = np.array(
            [
                np.cos(self.robot_angle + 3 * np.pi / 4),
                np.sin(self.robot_angle + 3 * np.pi / 4),
            ]
        )
        right = np.array(
            [
                np.cos(self.robot_angle - 3 * np.pi / 4),
                np.sin(self.robot_angle - 3 * np.pi / 4),
            ]
        )
        pts = [
            world_to_px(self.robot_pos + self.robot_radius * heading),
            world_to_px(self.robot_pos + self.robot_radius * left),
            world_to_px(self.robot_pos + self.robot_radius * right),
        ]

        # Change robot color to red if inside obstacle
        robot_color = (255, 0, 0) if self._check_collision() else (30, 144, 255)
        pygame.draw.polygon(surface, robot_color, pts)

        if self.render_mode == "human":
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(surface), axes=(1, 0, 2))

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
