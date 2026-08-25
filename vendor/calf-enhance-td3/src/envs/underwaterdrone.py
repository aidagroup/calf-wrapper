import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import Optional, Tuple, Dict, Any
from scipy.optimize import minimize_scalar
from src.utils.metrics_controller import MetricsCollector

TIME_STEP_SIZE = 0.02
DRONE_MASS = 1.0
DRONE_INERTIA = 0.1
DRONE_RADIUS = 0.2
DRAG_COEFF = 0.05
OFFSET_LAT = 0.2
MAX_F_LONG = 1.0
MAX_F_LAT = 0.5
GRAVITY = 0.5
TOP_Y = 4.0
HOLE_WIDTH = 4.0 * DRONE_RADIUS
MAX_X = 2.5
MAX_V = 3.0
MAX_OMEGA = 3.0


class UnderwaterDrone:
    def __init__(
        self,
        seed=None,
        random_generator=None,
        init_x=None,
        init_y=None,
        m=DRONE_MASS,
        I=DRONE_INERTIA,
        Cd=DRAG_COEFF,
        radius=DRONE_RADIUS,
        offset_lateral=OFFSET_LAT,
        gravity=GRAVITY,
        max_F_long=MAX_F_LONG,
        max_F_lat=MAX_F_LAT,
        top_y=TOP_Y,
        hole_width=HOLE_WIDTH,
    ):
        # Initialize random state generator with seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        elif random_generator is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = random_generator

        # Randomize initial state using the seeded generator
        self.x = self.rng.uniform(-MAX_X / 2, MAX_X / 2)
        self.y = self.rng.uniform(0, TOP_Y / 3.0)
        if init_x is not None:
            self.x = init_x
        if init_y is not None:
            self.y = init_y
        self.theta = self.rng.uniform(np.pi / 2 - np.pi / 20, np.pi / 2 + np.pi / 20)
        self.v_x = self.rng.uniform(-0.2, 0.2)
        self.v_y = self.rng.uniform(-0.2, 0.2)
        self.omega = self.rng.uniform(-0.2, 0.2)

        self.m = m
        self.I = I
        self.Cd = Cd
        self.radius = radius
        self.offset_lateral = offset_lateral
        self.gravity = gravity

        # Control bounds
        self.max_F_long = max_F_long  # max forward/backward thrust
        self.max_F_lat = max_F_lat  # max lateral thrust

        # To detect getting into air
        self.top_y = top_y
        self.hole_width = hole_width

        # We'll keep a "frozen" flag
        self.frozen = False

        # … everything as before up to preparing the drone nose …
        # shrink the nose triangle:
        self.nose_len = 0.17  # was 0.3
        self.nose_half_w = 0.07  # was 0.2
        self.nose_local_coords = np.array(
            [[self.nose_len, 0.0], [0.0, self.nose_half_w], [0.0, -self.nose_half_w]]
        )

    def step(self, action, dt=TIME_STEP_SIZE):
        """
        action = (F_long, F_lat)
          F_long: thrust in the drone's longitudinal direction (body x-axis).
          F_lat:  thrust in the drone's lateral direction (body y-axis).
        dt: time step for Euler integration.
          remembers the last action for thrust.
        """
        if self.frozen:
            return

        if self._in_hole():
            self._freeze()
            return
        self.last_action = action

        F_long, F_lat = action
        F_long = np.clip(F_long, -self.max_F_long, self.max_F_long)
        F_lat = np.clip(F_lat, -self.max_F_lat, self.max_F_lat)

        c = np.cos(self.theta)
        s = np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])

        thrust_body = np.array([F_long, F_lat])
        thrust_inertial = R @ thrust_body

        v = np.array([self.v_x, self.v_y])
        speed = np.linalg.norm(v)
        if speed > 1e-6:
            drag_dir = -v / speed
        else:
            drag_dir = np.array([0.0, 0.0])
        F_drag = self.Cd * speed**2 * drag_dir

        F_net = thrust_inertial + F_drag + np.array([0.0, -self.gravity])

        tau = self.offset_lateral * F_lat

        a_x = F_net[0] / self.m
        a_y = F_net[1] / self.m
        alpha = tau / self.I

        self.v_x += a_x * dt
        self.v_y += a_y * dt
        self.omega += alpha * dt

        self.x += self.v_x * dt
        self.y += self.v_y * dt
        self.theta += self.omega * dt

        self.x = np.clip(self.x, -MAX_X - 0.01, MAX_X + 0.01)
        self.y = np.clip(self.y, 0.0 - 0.01, TOP_Y + 0.01)

        if self._in_hole():
            self._freeze()
            return

        if self.y < 0.0:
            self.v_y = max(self.v_y, 0)
        if self.y > TOP_Y:
            self.v_y = min(self.v_y, 0)
        if self.x < -MAX_X:
            self.v_x = max(self.v_x, 0)
        if self.x > MAX_X:
            self.v_x = min(self.v_x, 0)

    def _in_hole(self):
        """
        Return True if the drone's center is at/above top_y
        AND within the horizontal hole region, i.e. x in [-hole_half, hole_half].
        """
        hole_half = self.hole_width / 2.0
        if self.y >= self.top_y:
            if -hole_half <= self.x <= hole_half:
                return True
        return False

    def _near_borders(self):
        """
        Return True if the drone's center is near the borders.
        """
        return (
            np.abs(self.x) > MAX_X - 0.01
            or self.y < 0.0 + 0.01
            or (
                np.abs(self.x) > self.hole_width / 2.0 + 0.01 and self.y >= TOP_Y - 0.01
            )
        )

    def _freeze(self):
        """Freeze the drone's motion. Zero velocities, 'frozen'=True."""
        self.v_x = 0.0
        self.v_y = 0.0
        self.omega = 0.0
        self.frozen = True

    def state(self):
        """Return (x, y, theta, v_x, v_y, omega)."""
        return np.array(
            [self.x, self.y, self.theta, self.v_x, self.v_y, self.omega],
            dtype=np.float32,
        )


class UnderwaterDroneEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": int(1.0 / TIME_STEP_SIZE),
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        init_x=None,
        init_y=None,
    ):
        # Define observation space
        # State is (x, y, theta, v_x, v_y, omega)
        self.observation_space = spaces.Box(
            low=np.array(
                [-np.inf, -np.inf, -1, -1, -np.inf, -np.inf, -np.inf], dtype=np.float32
            ),
            high=np.array(
                [np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf], dtype=np.float32
            ),
            dtype=np.float32,
        )
        self.rng = np.random.RandomState(seed)
        # Keep visual randomness isolated from episode-state randomness so
        # rendering/video capture cannot perturb future env resets.
        self.visual_rng = np.random.RandomState(seed)

        self.semimajor_axis = 0.9
        self.semiminor_axis = 0.6

        # Define action space
        # Action is (F_long, F_lat)
        self.action_space = spaces.Box(
            low=np.array([-MAX_F_LONG, -MAX_F_LAT], dtype=np.float32),
            high=np.array([MAX_F_LONG, MAX_F_LAT], dtype=np.float32),
            dtype=np.float32,
        )
        self.init_x = init_x
        self.init_y = init_y
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.drone = None

        # Display settings
        self.screen_width = 1200
        self.screen_height = 900
        self.scale_factor = 100  # Scale from physics units to pixels

        # Origin in screen coordinates (bottom-left in physics)
        self.origin_x = self.screen_width // 2
        self.origin_y = self.screen_height - 100

        # Counters
        self.n_near_borders = 0
        self.n_in_high_cost_area = 0
        self.n_resets = 0
        self.avoidance_score = -np.inf
        self.init_avoidance_score = None
        # Reset the environment

        # Trajectory & heatmap
        self.trajectory = []
        self.heatmap_surface = None

        # — Water‐line setup (Tank style, full width) —
        # rng seeded for reproducibility
        rng = np.random.RandomState(seed)

        # ten horizontal levels, jittered slightly
        y_lines = np.linspace(0.2, TOP_Y - 0.2, 10)
        y_lines += (rng.rand(len(y_lines)) - 0.5) * 0.2

        # build segments at each level
        self.water_segments = []
        for y in y_lines:
            segs = []
            for _ in range(5):
                length = rng.uniform(0.1, 0.3)  # each dash length
                start = rng.uniform(-MAX_X, MAX_X - length)
                segs.append((start, start + length))
            self.water_segments.append((y, segs))

        # drift speed for the lines
        self.water_drift_speed = 0.35

        # will be filled in _setup_rendering()
        self._water_lines = []

        # Shrink nose triangle
        self.nose_len = 0.12
        self.nose_half_w = 0.07
        self.nose_local_coords = np.array(
            [
                [self.nose_len, 0.0],
                [0.0, self.nose_half_w],
                [0.0, -self.nose_half_w],
            ]
        )

        # Prepare water‐line state immediately:
        self._water_lines = []
        for y, segs in self.water_segments:
            for xs, xe in segs:
                self._water_lines.append([y, xs, xe])

        # how much to crop left/right (15% default; increase to crop more)
        self.crop_frac = 0.18

        # cropping fractions for rgb_array output
        self.crop_top_frac = 0.20
        self.crop_bottom_frac = 0.09
        self.crop_left_frac = 0.28
        self.crop_right_frac = 0.28

        # Water‐surface wave parameters
        self.wave_amplitude = 5  # pixels of peak‐to‐peak wiggle
        self.wave_length = 200  # pixels between repeats
        self.wave_speed = 2.0  # cycles per second

        # High-quality heatmap settings
        self._heat_N = 255  # number of y-samples
        self._heat_M = 20  # number of gradient rings
        self._heat_max_alpha = 200
        self._heat_shrink = 0.91  # outer ring is 100%, inner is 90%

        # optional axes overlay
        self.show_axes = False
        self.axes_tick_step = 0.5

        # make sure font module is ready *before* we create the Font object
        pygame.font.init()  # <-- add
        self._font = pygame.font.SysFont(None, 16)

        self.show_target_icon = True  # set False if you want to hide the icon

        self.bubbles = []
        self._last_bubble_spawn = 0.0

        # ─── UnderwaterDroneEnv.__init__ ───
        self.last_action = np.zeros(2)  #  ←  add this anywhere in __init__

        # Finish init by resetting (match restore behavior: no seed here)
        self.reset()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.n_resets += 1

        if seed is not None:
            self.drone = UnderwaterDrone(
                seed=seed, init_x=self.init_x, init_y=self.init_y
            )
        elif self.rng is not None:
            self.drone = UnderwaterDrone(
                random_generator=self.rng, init_x=self.init_x, init_y=self.init_y
            )
        else:
            self.drone = UnderwaterDrone(init_x=self.init_x, init_y=self.init_y)

        # Clear trajectory history
        self.trajectory = []

        # Setup for rendering
        self._setup_rendering()
        self.n_near_borders = 0
        self.n_in_high_cost_area = 0
        self.avoidance_score = -np.inf
        self.init_avoidance_score = self._distance_to_exterior_of_high_cost_area(
            self.drone.x, self.drone.y
        )
        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------
    def set_axes(self, flag: bool = True, step: float | None = None):
        """Turn axes overlay on/off; optionally set tick step (physics units)."""
        self.show_axes = bool(flag)
        if step is not None:
            self.axes_tick_step = float(step)

    def _draw_water_wave(self):
        """Draw a small sine‐wave exactly at the hole y = TOP_Y."""
        y0 = self.to_pixels_y(TOP_Y)
        t = pygame.time.get_ticks() / 1000.0
        pts = []
        for px in range(0, self.screen_width + 1, 10):
            xw = (px - self.origin_x) / self.scale_factor
            theta = (
                2
                * np.pi
                * (xw * self.scale_factor / self.wave_length - self.wave_speed * t)
            )
            yoff = self.wave_amplitude * np.sin(theta)
            pts.append((px, int(y0 + yoff)))
        pygame.draw.lines(self.screen, (255, 255, 255, 120), False, pts, 2)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.last_action = action  #  ←  store action *before* physics

        # Execute drone physics
        self.drone.step(action, dt=TIME_STEP_SIZE)

        # Track trajectory
        self.trajectory.append((self.drone.x, self.drone.y))

        # Calculate reward
        reward = self._calculate_reward()

        terminated = False
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _is_in_high_cost_area(self):
        return (
            self.drone.x / self.semimajor_axis**2
            + (self.drone.y - TOP_Y / 2) ** 2 / self.semiminor_axis**2
            <= 1.0
        )

    def _calculate_reward(self) -> float:
        # Simple reward: -1 if frozen, otherwise 0.01 for each step plus height bonus
        return (
            -0.25 * ((self.drone.y - TOP_Y) ** 2)
            - 0.25 * (self.drone.x - 0.0) ** 2
            - 0.05 * self.drone.v_x**2
            - 0.05 * self.drone.v_y**2
            - 0.01 * self.drone.omega**2
            - 5 * (1 if self._is_in_high_cost_area() else 0)
        )

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.drone.x,
                self.drone.y,
                np.cos(self.drone.theta),
                np.sin(self.drone.theta),
                self.drone.v_x,
                self.drone.v_y,
                self.drone.omega,
            ],
            dtype=np.float32,
        )

    def _distance_to_parabola(self, x0, y0):
        """
        Robustly computes the shortest distance from point (x0, y0)
        to the parabola x = 0.81 - 2.25(y - 2)^2 using scalar minimization.

        Returns:
            distance: float - shortest distance
            x_closest: float - x on the parabola
            y_closest: float - y on the parabola
        """

        def D2(y):  # Squared distance function
            a, b = self.semimajor_axis**2, self.semiminor_axis**2
            x_parabola = a - a / b * (y - 2) ** 2
            return (x_parabola - x0) ** 2 + (y - y0) ** 2

        result = minimize_scalar(D2, bounds=(0, 4), method="bounded")
        distance = np.sqrt(result.fun)

        return distance

    def _distance_to_exterior_of_high_cost_area(self, x0, y0):
        """
        Computes the shortest distance from point (x0, y0)
        to the exterior of the high cost area.
        """
        if self._is_in_high_cost_area():
            return self._distance_to_parabola(x0, y0)
        else:
            return 0.0

    def _get_info(self) -> Dict[str, Any]:
        if self.drone._near_borders():
            self.n_near_borders += 1
        if self._is_in_high_cost_area():
            self.n_in_high_cost_area += 1

        current_avoidance_score = (
            self._distance_to_exterior_of_high_cost_area(self.drone.x, self.drone.y)
            - self.init_avoidance_score
        )
        self.avoidance_score = np.maximum(self.avoidance_score, current_avoidance_score)

        return {
            "x": self.drone.x,
            "y": self.drone.y,
            "is_frozen": self.drone.frozen,
            "is_in_hole": self.drone._in_hole(),
            "is_near_borders": self.drone._near_borders(),
            "n_near_borders": self.n_near_borders,
            "is_in_high_cost_area": self._is_in_high_cost_area(),
            "n_in_high_cost_area": self.n_in_high_cost_area,
            "current_avoidance_score": current_avoidance_score,
            "avoidance_score": np.copy(self.avoidance_score),
        }

    def _setup_rendering(self):
        if self.render_mode is None:
            return
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Underwater Drone Simulator")
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        grad = pygame.Surface((self.screen_width, self.screen_height))
        for py in range(self.screen_height):
            r = py / (self.screen_height - 1)
            if r < 0.2:
                # white → lightblue
                t, c0, c1 = r / 0.2, (255, 255, 255), (173, 216, 230)
            else:
                # lightblue → darkwater
                t, c0, c1 = (r - 0.2) / 0.8, (173, 216, 230), (25, 25, 150)
            color = tuple(int(c0[i] * (1 - t) + c1[i] * t) for i in range(3))
            grad.fill(color, (0, py, self.screen_width, 1))
        self.water_gradient_surface = grad
        # build the half-ellipse heatmap
        self._create_heatmap()

        # —————— Initialize water-line positions ——————
        self._water_lines = [
            [y, xs, xe] for y, segs in self.water_segments for xs, xe in segs
        ]

        if self.screen is None and self.render_mode in ["human", "rgb_array"]:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Underwater Drone Simulator")
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

            self.clock = pygame.time.Clock()

            # Create heatmap surface
            self._create_heatmap()

    def _create_heatmap(self):
        """Exact half-ellipse fade + animated bubbles + thick boundary + raft."""
        if self.screen is None:
            return

        a2 = self.semimajor_axis**2
        b2 = self.semiminor_axis**2
        b0 = TOP_Y / 2.0

        W, H = self.screen_width, self.screen_height
        surf = pygame.Surface((W, H), pygame.SRCALPHA)

        light = (14, 88, 24)
        dark = (0, 70, 0)

        for py in range(H):
            y = (self.origin_y - py) / self.scale_factor
            for px in range(W):
                x = (px - self.origin_x) / self.scale_factor
                val = x / a2 + ((y - b0) ** 2) / b2
                if val <= 1.0:
                    t = val  # fade from center to boundary
                    alpha = int(150 * (1.0 - t) * 0.6)  # 50% as opaque everywhere
                    # Blend color between light and dark green
                    r = int(light[0] * (1 - t) + dark[0] * t)
                    g = int(light[1] * (1 - t) + dark[1] * t)
                    b = int(light[2] * (1 - t) + dark[2] * t)
                    # Clamp
                    r = max(0, min(255, r))
                    g = max(0, min(255, g))
                    b = max(0, min(255, b))

                    alpha = max(0, min(255, alpha))
                    surf.set_at((px, py), (r, g, b, alpha))

        # ------------------------------------------------------------------
        #  Replace the old bubble-spawn block with **this**:

        self.bubbles = []  # clear previous list
        spawn_prob = 0.0005  # 1 % chance per pixel
        for py in range(H):
            y = (self.origin_y - py) / self.scale_factor
            for px in range(W):
                x = (px - self.origin_x) / self.scale_factor
                val = x / a2 + ((y - b0) ** 2) / b2
                if val <= 1.0 and self.visual_rng.random() < spawn_prob:
                    # centre is (x,y) which is *inside* the ellipse by construction
                    r_pix = self.visual_rng.randint(3, 7)  # radius 3–6 px
                    t_birth = self.visual_rng.uniform(0, 2)  # random phase
                    self.bubbles.append({"x": x, "y": y, "r": r_pix, "birth": t_birth})
        # ------------------------------------------------------------------

        # Raft at the hole using HOLE_WIDTH
        raft_w = int(HOLE_WIDTH * self.scale_factor)
        raft_h = int(0.05 * self.scale_factor)
        cx = self.to_pixels_x(0.0)
        cy = self.to_pixels_y(TOP_Y)
        raft = pygame.Rect(cx - raft_w // 2, cy, raft_w, raft_h)
        pygame.draw.rect(surf, (139, 69, 19), raft, border_radius=4)

        self.heatmap_surface = surf

    # -------------------------------------------------------------
    def _draw_target_icon(self):
        """Draw a small coloured flag at the hole to label the safe area."""
        if not self.show_target_icon:
            return
        cx = self.to_pixels_x(0.0)
        cy = self.to_pixels_y(TOP_Y)

        pole_h = 28
        pole_w = 3
        flag_w = 16
        flag_h = 12
        # pole
        pygame.draw.rect(self.screen, (80, 80, 80), (cx, cy - pole_h, pole_w, pole_h))
        # flag rectangle (bright orange)
        pygame.draw.rect(
            self.screen, (255, 140, 0), (cx + pole_w, cy - pole_h + 2, flag_w, flag_h)
        )
        # black border around flag
        pygame.draw.rect(
            self.screen, (0, 0, 0), (cx + pole_w, cy - pole_h + 2, flag_w, flag_h), 1
        )

    def _draw_thrust_arrows(self):
        """Draw thrust force arrows for current control on the drone."""
        # Get drone center and orientation
        cx = self.to_pixels_x(self.drone.x)
        cy = self.to_pixels_y(self.drone.y)
        angle = self.drone.theta

        # Scale for visual length
        scale = 40

        # Most recent action (or zero if unavailable)
        if hasattr(self, "last_action"):
            F_long, F_lat = self.last_action
        else:
            F_long, F_lat = 0.0, 0.0

        # Body frame vectors
        dx_long = scale * F_long * np.cos(angle)
        dy_long = -scale * F_long * np.sin(angle)
        dx_lat = -scale * F_lat * np.sin(angle)
        dy_lat = -scale * F_lat * np.cos(angle)

        # Draw arrows (longitudinal: red, lateral: blue)
        end_long = (int(cx + dx_long), int(cy + dy_long))
        end_lat = (int(cx + dx_lat), int(cy + dy_lat))
        pygame.draw.line(self.screen, (255, 60, 60), (cx, cy), end_long, 4)
        pygame.draw.line(self.screen, (60, 60, 255), (cx, cy), end_lat, 4)
        # Optionally: draw arrowheads here

    def _update_water_lines(self, dt):
        new_lines = []
        for y, xs, xe in self._water_lines:
            xs2 = xs + self.water_drift_speed * dt
            xe2 = xe + self.water_drift_speed * dt

            # wrap when the entire dash has left the right wall
            if xs2 >= MAX_X:
                xs2 -= 2 * MAX_X  # jump back to the left wall
                xe2 -= 2 * MAX_X
            # draw
            px1 = self.to_pixels_x(xs2)
            py = self.to_pixels_y(y)
            px2 = self.to_pixels_x(xe2)
            pygame.draw.line(self.screen, (255, 255, 255, 120), (px1, py), (px2, py), 2)
            new_lines.append([y, xs2, xe2])
        self._water_lines = new_lines

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self.screen is None:
            self._setup_rendering()

        # 1) Draw the vertical water gradient
        self.screen.blit(self.water_gradient_surface, (0, 0))

        # 2) Water lines under surface
        self._update_water_lines(TIME_STEP_SIZE)

        # 3) Gaussian obstacle
        if self.heatmap_surface is not None:
            self.screen.blit(self.heatmap_surface, (0, 0))

        # 4) Hole halo & surface wiggle
        self._draw_target_icon()
        self._draw_water_wave()

        # 5) Trajectory
        if len(self.trajectory) > 1:
            pts = [
                (self.to_pixels_x(x), self.to_pixels_y(y)) for x, y in self.trajectory
            ]
            pygame.draw.lines(self.screen, (5, 5, 0), False, pts, 2)
            for i, p in enumerate(pts):
                if i % 3 == 0:
                    pygame.draw.circle(self.screen, (255, 255, 255, 120), p, 1)

        # 6) Drone body & protruding nose
        pygame.draw.circle(
            self.screen,
            (0, 180, 180),
            (self.to_pixels_x(self.drone.x), self.to_pixels_y(self.drone.y)),
            int(self.drone.radius * self.scale_factor),
        )
        c, s = np.cos(self.drone.theta), np.sin(self.drone.theta)
        R = np.array([[c, -s], [s, c]])
        tri = (R @ self.drone.nose_local_coords.T).T + [self.drone.x, self.drone.y]
        tri_px = [(self.to_pixels_x(x), self.to_pixels_y(y)) for x, y in tri]
        pygame.draw.polygon(self.screen, (139, 0, 0), tri_px)

        # Draw drone with thick black outline for visibility

        drone_px = self.to_pixels_x(self.drone.x)
        drone_py = self.to_pixels_y(self.drone.y)
        radius_px = int(self.drone.radius * self.scale_factor)
        pygame.draw.circle(
            self.screen, (0, 0, 0), (drone_px, drone_py), radius_px + 3
        )  # thick black border
        pygame.draw.circle(self.screen, (0, 180, 180), (drone_px, drone_py), radius_px)

        # In your render() function, call this after drawing the drone body:
        if hasattr(self, "last_action"):
            self._draw_thrust_arrows()

            # --- Animate and draw bubbles on top of the heatmap ---
        if hasattr(self, "bubbles"):
            t_now = pygame.time.get_ticks() / 1000.0
            for bub in self.bubbles:
                # Bubbles rise a bit and fade out after 2 seconds
                t = (t_now - bub["birth"]) % 2.0
                fade = max(0, 1 - t / 2.0)
                px = self.to_pixels_x(bub["x"])
                py = self.to_pixels_y(bub["y"] + t * 0.12)
                alpha = int(120 * fade)
                color = (128, 200, 128, alpha)  # purplish-green
                pygame.draw.circle(self.screen, color, (px, py), bub["r"])

        # # 6½) axes with ticks / values --------------------------------------
        # if self.show_axes:
        #     # axis lines
        #     pygame.draw.line(
        #         self.screen,
        #         (0, 0, 0),
        #         (self.to_pixels_x(-MAX_X), self.to_pixels_y(0)),
        #         (self.to_pixels_x(MAX_X), self.to_pixels_y(0)),
        #         1,
        #     )
        #     pygame.draw.line(
        #         self.screen,
        #         (0, 0, 0),
        #         (self.to_pixels_x(0), self.to_pixels_y(0)),
        #         (self.to_pixels_x(0), self.to_pixels_y(TOP_Y)),
        #         1,
        #     )

        #     # ticks & numeric labels
        #     step = self.axes_tick_step
        #     # X-axis ticks every 'step' units
        #     x_vals = np.arange(-MAX_X, MAX_X + 1e-6, step)
        #     for x in x_vals:
        #         px = self.to_pixels_x(x)
        #         py = self.to_pixels_y(0)
        #         pygame.draw.line(self.screen, (0, 0, 0), (px, py - 3), (px, py + 3), 1)
        #         lbl = self._font.render(f"{x:.1f}", True, (0, 0, 0))
        #         self.screen.blit(lbl, lbl.get_rect(center=(px, py + 12)))
        #     # Y-axis ticks
        #     y_vals = np.arange(0, TOP_Y + 1e-6, step)
        #     for y in y_vals:
        #         px = self.to_pixels_x(0)
        #         py = self.to_pixels_y(y)
        #         pygame.draw.line(self.screen, (0, 0, 0), (px - 3, py), (px + 3, py), 1)
        #         lbl = self._font.render(f"{y:.1f}", True, (0, 0, 0))
        #         self.screen.blit(lbl, lbl.get_rect(center=(px - 18, py)))

        # 7) Output: human or cropped rgb_array
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            full_surf = self.screen  # window-sized surface
            h, w = self.screen_height, self.screen_width

            # use the fractions you already store on the object
            top = int(self.crop_top_frac * h)
            bottom = h - int(self.crop_bottom_frac * h)
            left = int(self.crop_left_frac * w)
            right = w - int(self.crop_right_frac * w)

            # grab the cropped view (no copy) and convert to NumPy
            region = full_surf.subsurface(
                pygame.Rect(left, top, right - left, bottom - top)
            )

            # no re-scaling → no distortion, full native resolution
            return pygame.surfarray.array3d(region).transpose(1, 0, 2)

    def to_pixels_x(self, x: float) -> int:
        return int(self.origin_x + x * self.scale_factor)

    def to_pixels_y(self, y: float) -> int:
        return int(self.origin_y - y * self.scale_factor)

    def close(self):
        if self.screen:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


class UnderwaterDroneMetricsCollector(MetricsCollector):
    def __init__(self, rolling_window_size: int = 20):
        super().__init__()
        self.rolling_window_size = rolling_window_size

    def collect_metrics_from_final_episode_info(self, info: dict, step: int):
        super().collect_metrics_from_final_episode_info(info, step)
        self.append_metric("episode_stats/is_in_hole", info["is_in_hole"], step=step)
        self.rolling_window["is_in_hole"].append(info["is_in_hole"])
        self.append_metric(
            f"episode_stats/is_in_hole_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["is_in_hole"]),
            step=step,
        )
        self.append_metric(
            "episode_stats/avoidance_score", info["avoidance_score"], step=step
        )
        self.rolling_window["avoidance_score"].append(info["avoidance_score"])
        self.append_metric(
            f"episode_stats/avoidance_score_rolling_{self.rolling_window_size}",
            np.mean(self.rolling_window["avoidance_score"]),
            step=step,
        )
