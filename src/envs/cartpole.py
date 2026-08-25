# Modified from gymnasium.envs.classic_control.cartpole

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class CartPoleSwingupEnv(gym.Env[np.ndarray, int | np.ndarray]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int | None = None,
        terminate_on_out_of_bounds: bool = True,
        saturate_state_on_out_of_bounds: bool = False,
        reward_position_clip: float | None = None,
        position_termination_threshold: float = 5.0,
        velocity_termination_threshold: float = 8.0,
        angular_velocity_termination_threshold: float = 10.0,
    ):
        super().__init__()

        if reward_position_clip is not None and reward_position_clip <= 0:
            raise ValueError("reward_position_clip must be positive")
        for name, value in (
            ("position_termination_threshold", position_termination_threshold),
            ("velocity_termination_threshold", velocity_termination_threshold),
            (
                "angular_velocity_termination_threshold",
                angular_velocity_termination_threshold,
            ),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.gravconst = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length

        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                np.finfo(np.float32).max,  # x
                np.finfo(np.float32).max,  # x_dot
                1,  # cos_theta
                1,  # sin_theta
                np.finfo(np.float32).max,  # theta_dot
            ],
            dtype=np.float32,
        )
        self.force_mag = 10.0
        self.action_space = spaces.Box(-self.force_mag, self.force_mag, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.terminate_on_out_of_bounds = terminate_on_out_of_bounds
        self.saturate_state_on_out_of_bounds = saturate_state_on_out_of_bounds
        self.reward_position_clip = reward_position_clip
        self.position_termination_threshold = position_termination_threshold
        self.velocity_termination_threshold = velocity_termination_threshold
        self.angular_velocity_termination_threshold = angular_velocity_termination_threshold

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = np.clip(action, -self.force_mag, self.force_mag).reshape(-1).item()
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravconst * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        if self.saturate_state_on_out_of_bounds:
            clipped_x = float(
                np.clip(
                    x,
                    -self.position_termination_threshold,
                    self.position_termination_threshold,
                )
            )
            if clipped_x != x and np.sign(x_dot) == np.sign(x):
                x_dot = 0.0
            x = clipped_x
            x_dot = float(
                np.clip(
                    x_dot,
                    -self.velocity_termination_threshold,
                    self.velocity_termination_threshold,
                )
            )
            theta_dot = float(
                np.clip(
                    theta_dot,
                    -self.angular_velocity_termination_threshold,
                    self.angular_velocity_termination_threshold,
                )
            )

        self.state = (x, x_dot, theta, theta_dot)

        terminated = self.terminate_on_out_of_bounds and bool(
            abs(x) > self.position_termination_threshold
            or abs(theta_dot) > self.angular_velocity_termination_threshold
            or abs(x_dot) > self.velocity_termination_threshold
        )

        if not terminated:
            reward_x = (
                float(np.clip(x, -self.reward_position_clip, self.reward_position_clip))
                if self.reward_position_clip is not None
                else x
            )
            reward = (
                -0.5 * angle_normalize(theta) ** 2
                - 0.5 * reward_x**2
                - 0.05 * theta_dot**2
                - 0.05 * x_dot**2
            )
        else:
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return np.array(
            [
                self.state[0],
                self.state[1],
                np.cos(self.state[2]),
                np.sin(self.state[2]),
                self.state[3],
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        state_init = np.array([0, 0, np.pi, 0], dtype=np.float32)
        high = np.array([1, 1, np.pi, 1], dtype=np.float32)
        self.state = state_init + self.np_random.uniform(low=-high, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        left, right, top, bottom = (
            -cartwidth / 2,
            cartwidth / 2,
            cartheight / 2,
            -cartheight / 2,
        )
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [
            (left, bottom),
            (left, top),
            (right, top),
            (right, bottom),
        ]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        left, right, top, bottom = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [
            (left, bottom),
            (left, top),
            (right, top),
            (right, bottom),
        ]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
