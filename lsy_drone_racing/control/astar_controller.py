import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from crazyflow.sim.visualize import draw_line, draw_points

from lsy_drone_racing.control.astar import astar_3d
from lsy_drone_racing.control import Controller

class PIDTracker:
    """Per-axis PID that outputs a correction in that axis."""
    def __init__(self, kp, ki, kd, max_integral=2.0, max_output=5.0):
        self.kp           = kp
        self.ki           = ki
        self.kd           = kd
        self.max_integral = max_integral
        self.max_output   = max_output
        self._integral    = np.zeros(3)
        self._last_error  = np.zeros(3)
        self._first       = True

    def reset(self):
        self._integral   = np.zeros(3)
        self._last_error = np.zeros(3)
        self._first      = True

    def update(self, error: np.ndarray, dt: float) -> np.ndarray:
        self._integral += error * dt
        self._integral  = np.clip(self._integral, -self.max_integral, self.max_integral)
        if self._first:
            derivative  = np.zeros(3)
            self._first = False
        else:
            derivative = (error - self._last_error) / max(dt, 1e-6)
        self._last_error = error.copy()
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return np.clip(output, -self.max_output, self.max_output)

class AstarController(Controller):
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)

        self._freq     = config.env.freq
        self._tick     = 0
        self._finished = False
        self.dt        = 1.0 / config.env.freq

        self.cruise_speed    = 0.4 
        self.look_ahead_dist = 0.0

        self.current_s   = 0.0
        self._arc_length = 1.0
        self._t_total    = 1.0

        self.obstacle_radius   = 0.1
        self.detour_margin     = 0.22
        self.gate_offset       = 0.22
        self.gate_half_opening = 0.22
        self.max_obstacle_dist = 2
                                      
        self._gate_corners = []

        self.pid_pos = PIDTracker(0.5, 0.01, 0.40, max_integral=1.0, max_output=3.0)
        self.pid_vel = PIDTracker(0.25, 0.01, 0.1, max_integral=0.5, max_output=2.0)
        self.pid_acc = PIDTracker(0.01, 0.00, 0.0, max_integral=0.2, max_output=1.0)

        self._last_gates_pos     = None
        self._last_gates_quat    = None
        self._last_obstacles_pos = None
        self._last_target_gate   = -2

        self.voxel_size = 0.1


        self._build_spline(obs)


    def _build_arc_length_table(self, n_samples: int = 500):
        t_samp   = np.linspace(0, self._t_total, n_samples)
        pts      = self.spline(t_samp)
        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc_s    = np.concatenate([[0.0], np.cumsum(seg_lens)])

        self._arc_length = float(arc_s[-1])
        self._s_to_t     = CubicSpline(arc_s, t_samp)

    def _s_to_spline(self, s: float) -> float:
        s = float(np.clip(s, 0.0, self._arc_length))
        return float(np.clip(float(self._s_to_t(s)), 0.0, self._t_total))


    def _gate_frame_obstacles(self, gates_pos, gates_quat):
        virtual_obs = []
        self._gate_corners = []
        GATE_INNER_HALF = 0.20
        GATE_OUTER_HALF = 0.4

        for i in range(len(gates_pos)):
            r       = R.from_quat(gates_quat[i])
            lateral = r.apply([0, 1, 0])
            lateral = np.array([lateral[0], lateral[1], 0.0])
            lateral /= max(np.linalg.norm(lateral), 1e-6)

            for half in (GATE_INNER_HALF, GATE_OUTER_HALF):
                corners = []
                for lat_sign in (+1, -1):
                    for z_sign in (+1, -1):
                        corner = (gates_pos[i]
                                + lat_sign * half * lateral
                                + np.array([0, 0, z_sign * half]))
                        if half == GATE_OUTER_HALF:
                            corners.append(corner)
                        virtual_obs.append(corner)
                        self._gate_corners.append(corner)

                if half == GATE_OUTER_HALF:
                    edges = [
                        (corners[0], corners[2]),  # top:    +z edge
                        (corners[1], corners[3]),  # bottom: -z edge
                        (corners[0], corners[1]),  # right:  +lat edge
                        (corners[2], corners[3]),  # left:   -lat edge
                    ]

                    for a, b in edges:
                        for t in (0.2, 0.4, 0.6, 0.8):
                            pt = a + t * (b - a)
                            virtual_obs.append(pt)
                            self._gate_corners.append(pt)

        return virtual_obs

    def _build_spline(self, obs):
        start_pos  = obs["pos"]
        gates_pos  = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
        obstacles  = obs["obstacles_pos"]

        target_gate = int(obs["target_gate"])


        remaining_pos  = gates_pos[target_gate:]
        remaining_quat = gates_quat[target_gate:]

        gate_points = self._gate_frame_obstacles(gates_pos, gates_quat)


        sampled_rods = []
        ROD_MAX_HEIGHT = 2.0
        ROD_STEP = 0.20

        for rod_pos in obstacles:
            zs = np.arange(0.0, ROD_MAX_HEIGHT + ROD_STEP, ROD_STEP)
            for z in zs:
                sampled_rods.append(np.array([rod_pos[0], rod_pos[1], z]))

        all_obstacles = gate_points + sampled_rods

        raw_pre_gate_waypoints = []
        raw_waypoints = []
        raw_post_gate_waypoints = []
        gate_normals = []

        for i in range(len(remaining_pos)):
            r = R.from_quat(remaining_quat[i])
            gate_normal = r.apply([1, 0, 0])

            pre_wp  = remaining_pos[i] - gate_normal * self.gate_offset
            post_wp = remaining_pos[i] + gate_normal * self.gate_offset

            raw_pre_gate_waypoints.append(pre_wp)
            raw_waypoints.append(remaining_pos[i].copy())
            raw_post_gate_waypoints.append(post_wp)
            gate_normals.append(gate_normal)


        final_waypoints = []

        for j, point in enumerate(astar_3d(
            start = start_pos,
            goal = raw_pre_gate_waypoints[0],
            obstacles=all_obstacles,
            voxel_size=self.voxel_size,
            obstacle_clearance=self.detour_margin,
            gate_normal= (None, gate_normals[0]),
        )):
            if j % 3 == 0:
                final_waypoints.append(point)
        final_waypoints.append(raw_waypoints[0])



        for i in range(1, len(remaining_pos)):
            start = raw_post_gate_waypoints[i-1]
            end_pos   = raw_pre_gate_waypoints[i]
            for k, point in enumerate(astar_3d(
                start=start,
                goal=end_pos,
                obstacles=all_obstacles,
                voxel_size=self.voxel_size,
                obstacle_clearance=self.detour_margin,
                gate_normal= (gate_normals[i-1], None) if i == len(remaining_pos) else (gate_normals[i-1], gate_normals[i]),
            )):
                if k % 3 == 0:
                    final_waypoints.append(point)
            final_waypoints.append(raw_waypoints[i])
        final_waypoints.append(raw_waypoints[-1])
        final_waypoints.append(raw_post_gate_waypoints[-1])

        waypoints     = np.vstack(final_waypoints)
        self._t_total = len(waypoints) - 1
        t_steps       = np.arange(len(waypoints))


        current_vel = np.array(obs.get("vel", [0.0, 0.0, 0.0]))
        speed = np.linalg.norm(current_vel)

        if speed > 0.1:
            dist_to_next = np.linalg.norm(waypoints[1] - waypoints[0])
            start_tangent = (current_vel / speed) * dist_to_next
            self.spline = CubicSpline(t_steps, waypoints, bc_type=((1, start_tangent), 'natural'))
        else:
            self.spline = CubicSpline(t_steps, waypoints, bc_type="natural")

        self._build_arc_length_table()
        self._visual_trajectory = self.spline(np.linspace(0, self._t_total, 800))



    def _state_changed(self, obs):
        if self._last_gates_pos is None:
            return True

        target_gate = int(obs["target_gate"])

        if target_gate >= 0:
            if not np.allclose(
                obs["gates_pos"][target_gate],
                self._last_gates_pos[target_gate],
                atol=1e-2,
            ):
                return True

        if target_gate >= 0:
            target_gate_pos = obs["gates_pos"][target_gate]
            for i, obs_pos in enumerate(obs["obstacles_pos"]):
                if np.linalg.norm(target_gate_pos - obs_pos) > self.max_obstacle_dist:
                    continue  # not near the target gate — ignore
                if not np.allclose(obs_pos, self._last_obstacles_pos[i], atol=0.001):
                    return True

        return False

    def _cache_state(self, obs):
        self._last_gates_pos     = obs["gates_pos"].copy()
        self._last_gates_quat    = obs["gates_quat"].copy()
        self._last_obstacles_pos = obs["obstacles_pos"].copy()
        self._last_target_gate   = int(obs["target_gate"])


    def compute_control(self, obs, info=None):
        if self._state_changed(obs):
            actual_pos_before = np.array(obs["pos"])
            try:
                self._build_spline(obs)
                self._cache_state(obs)

                s_search = np.linspace(0.0, min(self._arc_length, self.cruise_speed * 3.0), 200)
                t_search = np.array([self._s_to_spline(s) for s in s_search])
                pts      = self.spline(t_search)
                dists    = np.linalg.norm(pts - actual_pos_before, axis=1)
                self.current_s = float(s_search[np.argmin(dists)])

            except Exception as e:
                pass

        target_s = min(self.current_s + self.look_ahead_dist, self._arc_length)
        t        = self._s_to_spline(target_s)

        ref_pos        = self.spline(t)
        spline_tangent = self.spline(t, 1)
        ds_dt          = max(np.linalg.norm(spline_tangent), 1e-6)
        dt_ds          = 1.0 / ds_dt

        ref_vel = spline_tangent    * dt_ds    * self.cruise_speed
        ref_acc = self.spline(t, 2) * dt_ds**2 * self.cruise_speed**2

        actual_pos = np.array(obs["pos"])
        actual_vel = np.array(obs["vel"])

        pos_correction = self.pid_pos.update(ref_pos - actual_pos, self.dt)
        vel_correction = self.pid_vel.update(ref_vel - actual_vel, self.dt)
        acc_correction = self.pid_acc.update(ref_acc,              self.dt)

        return np.array([
            *(ref_pos + pos_correction),
            *(ref_vel + vel_correction),
            *(ref_acc + acc_correction),
            0,
            0, 0, 0,
        ], dtype=np.float32)



    def step_callback(self, action, obs, reward, terminated, truncated, info):
        drone_pos = np.array(obs["pos"])

        s_search = np.linspace(
            max(0.0,              self.current_s - 0.2),
            min(self._arc_length, self.current_s + self.cruise_speed * 0.5),
            60,
        )
        t_search = np.array([self._s_to_spline(s) for s in s_search])
        pts      = self.spline(t_search)
        dists    = np.linalg.norm(pts - drone_pos, axis=1)
        best_s   = float(s_search[np.argmin(dists)])

        time_advance   = self.cruise_speed * self.dt
        self.current_s = float(np.clip(
            max(self.current_s + time_advance * 0.5, best_s),
            0.0, self._arc_length,
        ))

        if int(obs["target_gate"]) == -1:
            self._finished = True

        self._tick += 1
        return self._finished

    def episode_reset(self):
        self._tick     = 0
        self._finished = False
        self.current_s = 0.0
        self.pid_pos.reset()
        self.pid_vel.reset()
        self.pid_acc.reset()
        self._last_gates_pos     = None
        self._last_gates_quat    = None
        self._last_obstacles_pos = None
        self._last_target_gate   = -2


    def render_callback(self, sim):
        draw_line(sim, self._visual_trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        t_now    = self._s_to_spline(
            min(self.current_s + self.look_ahead_dist, self._arc_length)
        )
        setpoint = self.spline(t_now).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

        if self._gate_corners:
            corners = np.array(self._gate_corners)
            draw_points(sim, corners, rgba=(1.0, 0.5, 0.0, 1.0), size=0.03)