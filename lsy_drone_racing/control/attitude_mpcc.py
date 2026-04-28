"""
MPCC Controller for Drone Racing

Features:
- Model Predictive Contouring Control (MPCC) using acados
- Modular path planning via path module
- Dynamic replanning when environment changes
- Configurable speed/stability trade-offs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat, dot, DM, norm_2, floor, if_else
from crazyflow.sim.visualize import draw_line
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

# Import path planning module
from lsy_drone_racing.control.path import PathPlanner, PathConfig

# Import drone racing framework
from drone_models.core import load_params
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MPCCConfig:
    """Configuration for MPCC controller."""
    # MPC Horizon
    N_horizon: int = 40                    # Number of horizon steps
    T_horizon: float = 0.7                 # Horizon time (seconds)
    
    # Arc-length model
    model_arc_step: float = 0.05            # Arc length discretization
    model_traj_length: float = 15.0         # Trajectory length in model
    
    # Cost function weights (tunable for speed/stability trade-off)
    # Higher values = more tracking accuracy (stability)
    # Lower values = more speed
    q_lag: float = 80.0                    # Lag error weight
    q_lag_peak: float = 500.0              # Lag error weight at gates
    q_contour: float =120.0               # Contour error weight
    q_contour_peak: float = 700.0          # Contour error weight at gates
    q_attitude: float = 1.0                # Attitude regularization
    
    # Control smoothness
    r_thrust: float = 0.2                  # Thrust rate penalty
    r_roll: float = 0.3                    # Roll rate penalty
    r_pitch: float = 0.3                   # Pitch rate penalty
    r_yaw: float = 0.5                    # Yaw rate penalty
    
    # Speed incentive
    mu_speed: float = 10.0                  # Progress reward
    w_speed_gate: float = 10.0               # Speed penalty at gates
    
    #Cost weight multipliers
    k_qc_gate: float = 0.6                   # Cost weight multiplier near gates
    k_qc_obs: float = 0.7                    # Cost weight multiplier near obstacles


    # Safety bounds
    pos_bounds: tuple = (
        (-2.5, 2.5),                        # X bounds
        (-2.0, 1.5),                        # Y bounds
        (-0.1, 2.0),                        # Z bounds
    )
    vel_bounds: tuple = (-1.0, 4.0)         # Velocity bounds (m/s)
    
    # Path planning
    planned_duration: float = 30.0          # Nominal trajectory duration

    # Theta projection / branch protection
    theta_proj_window_back: float = 0.8     # Local projection window behind current theta
    theta_proj_window_front: float = 2.5    # Local projection window ahead of current theta
    theta_proj_sample_step: float = 0.03    # Sampling step for local projection search

    # Gate-order stage tracking
    gate_stage_theta_margin: float = 0.20   # Extra theta margin beyond active gate
    gate_stage_slowdown_window: float = 0.35  # Slowdown window near stage end
    v_theta_stage_scale_min: float = 0.35   # Minimum scale for v_theta_cmd upper bound near stage end
    
    # Logging settings
    log_interval: int = 100                 # Print debug info every N ticks


# =============================================================================
# MPCC Controller
# =============================================================================

class MPCCController(Controller):
    """
    Model Predictive Contouring Control for Drone Racing.
    
    This controller optimizes both tracking accuracy and progress speed
    along a pre-planned trajectory using nonlinear MPC.
    """
    
    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
        mpcc_config: Optional[MPCCConfig] = None,
        path_config: Optional[PathConfig] = None
    ):
        """
        Initialize the MPCC controller.
        
        Args:
            obs: Initial observation.
            info: Initial environment info.
            config: Race configuration.
            mpcc_config: MPCC configuration. Uses defaults if None.
            path_config: Path planning configuration. Uses defaults if None.
        """
        super().__init__(obs, info, config)
        
        # Configurations
        self.mpcc_cfg = mpcc_config or MPCCConfig()
        self.path_cfg = path_config or PathConfig()
        
        # Controller state
        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self.finished = False
        
        # Load dynamics parameters
        self._dyn_params = load_params("so_rpy_rotor", config.sim.drone_model)
        self._mass = float(self._dyn_params["mass"])
        self._gravity = -float(self._dyn_params["gravity_vec"][-1])
        self.hover_thrust = self._mass * self._gravity
        
        # Initialize path planner
        self.path_planner = PathPlanner(self.path_cfg)
        
        # Store initial position
        self._initial_pos = obs["pos"].copy()
        
        # Environment change detection
        self._last_gate_flags = None
        self._last_obst_flags = None
        
        # Gate detection tracking
        num_gates = len(obs['gates_pos'])
        self._gate_detected_flags = np.zeros(num_gates, dtype=bool)
        self._gate_real_positions = np.full((num_gates, 3), np.nan)
        self._active_gate_stage = 0
        self._stage_theta_end = np.inf
        
        # Plan initial trajectory
        self._plan_trajectory(obs)
        
        # MPC parameters
        self.N = self.mpcc_cfg.N_horizon
        self.T = self.mpcc_cfg.T_horizon
        self.dt = self.T / self.N
        self.model_arc_step = self.mpcc_cfg.model_arc_step
        self.model_traj_length = self.mpcc_cfg.model_traj_length
        
        # Build MPCC solver
        self._build_solver()
        
        # Initialize control states
        self.last_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        
        # Current observation (for debug)
        self._current_pos = obs["pos"].copy()
        self._actual_path_points = [obs["pos"].copy()]
        print(f"[MPCC] Initialized. Horizon: N={self.N}, T={self.T:.2f}s")
        print(f"[MPCC] Arc trajectory length: {self.arc_trajectory.x[-1]:.2f}")
    
    # =========================================================================
    # Trajectory Planning
    # =========================================================================
    
    def _plan_trajectory(self, obs: dict[str, NDArray[np.floating]]):
        """Plan or replan the trajectory."""
        print(f"[MPCC] Planning trajectory at T={self._step_count / self._ctrl_freq:.2f}s")
    
        obs_planning = obs.copy()
        obs_planning['pos'] = self._initial_pos.copy()
            
        # Use path planner to generate trajectory
        result = self.path_planner.plan_trajectory(
            obs_planning,
            trajectory_duration=self.mpcc_cfg.planned_duration,
            sampling_freq=self._ctrl_freq,
            for_mpcc=True,
            mpcc_extension_length=self.mpcc_cfg.model_traj_length
        )
        
        # Store full result
        self._trajectory_result = result
        
        # Store results
        self.trajectory = result.spline
        self.arc_trajectory = result.arc_spline
        self.waypoints = result.waypoints
        self.total_arc_length = result.total_length
        self._cache_planned_path()
        self._refresh_gate_stage(obs)
        
        # Cache for cost computation
        self._cached_gate_centers = obs["gates_pos"].copy()
        self._cached_obstacles = obs["obstacles_pos"].copy()

    def _cache_planned_path(self):
        """Cache a uniformly sampled version of the full planned arc-length path."""
        if not hasattr(self, "arc_trajectory"):
            self._planned_path_points = np.zeros((0, 3), dtype=float)
            return
        n_samples = 300
        theta = np.linspace(0.0, float(self.arc_trajectory.x[-1]), n_samples)
        self._planned_path_points = np.asarray(self.arc_trajectory(theta), dtype=float)

    def _refresh_gate_stage(self, obs: dict[str, NDArray[np.floating]]):
        """Refresh active gate stage and corresponding theta endpoint."""
        if self._trajectory_result.gate_thetas is None or len(self._trajectory_result.gate_thetas) == 0:
            self._active_gate_stage = 0
            self._stage_theta_end = float(self.arc_trajectory.x[-1])
            return

        visited = np.array(obs.get("gates_visited", []), dtype=bool)
        if visited.size == 0:
            next_gate = 0
        else:
            next_gate = int(np.argmax(~visited)) if np.any(~visited) else int(len(self._trajectory_result.gate_thetas))

        self._active_gate_stage = min(next_gate, len(self._trajectory_result.gate_thetas))
        if self._active_gate_stage >= len(self._trajectory_result.gate_thetas):
            self._stage_theta_end = float(self.arc_trajectory.x[-1])
        else:
            self._stage_theta_end = min(
                float(self.arc_trajectory.x[-1]),
                float(self._trajectory_result.gate_thetas[self._active_gate_stage]) + self.mpcc_cfg.gate_stage_theta_margin,
            )

    def _project_theta_local(self, position: NDArray[np.floating], center_theta: Optional[float] = None) -> float:
        """
        Project position to trajectory using a local theta window.

        This avoids jumping to nearby parallel branches in self-intersections.
        """
        if not hasattr(self, "arc_trajectory"):
            return 0.0

        cfg = self.mpcc_cfg
        theta_max = float(self.arc_trajectory.x[-1])
        center = float(self.last_theta if center_theta is None else center_theta)

        lo = max(0.0, center - cfg.theta_proj_window_back)
        hi = min(theta_max, center + cfg.theta_proj_window_front)
        hi = min(hi, float(self._stage_theta_end))
        if hi <= lo + 1e-6:
            return float(np.clip(center, 0.0, theta_max))

        step = max(1e-3, float(cfg.theta_proj_sample_step))
        theta_samples = np.arange(lo, hi + step, step)
        pts = np.asarray(self.arc_trajectory(theta_samples), dtype=float)
        dists = np.linalg.norm(pts - position[None, :], axis=1)
        idx = int(np.argmin(dists))
        return float(theta_samples[idx])

    def _apply_stage_progress_limit(self, obs: dict[str, NDArray[np.floating]]):
        """Update active stage and scale v_theta upper bound near stage end."""
        self._refresh_gate_stage(obs)

        base_ub = 1.8
        remain = max(0.0, float(self._stage_theta_end) - float(self.last_theta))
        window = max(1e-3, float(self.mpcc_cfg.gate_stage_slowdown_window))
        if remain >= window:
            scale = 1.0
        else:
            min_scale = float(self.mpcc_cfg.v_theta_stage_scale_min)
            scale = min_scale + (1.0 - min_scale) * (remain / window)

        v_theta_ub = max(0.2, base_ub * scale)
        ubu_stage = np.array([10.0, 10.0, 10.0, 10.0, v_theta_ub], dtype=float)
        lbu_stage = np.array([-10.0, -10.0, -10.0, -10.0, 0.0], dtype=float)
        for i in range(self.N):
            self.solver.constraints_set(i, "lbu", lbu_stage)
            self.solver.constraints_set(i, "ubu", ubu_stage)

    @staticmethod
    def _downsample_polyline(points: NDArray[np.floating], max_points: int) -> NDArray[np.floating]:
        """Downsample a polyline to keep rendering cost bounded."""
        if points.shape[0] <= max_points:
            return points
        idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=int)
        return points[idx]
    
    # =========================================================================
    # MPCC Solver Construction
    # =========================================================================
    
    def _build_solver(self):
        """Build the acados MPCC solver."""
        # Build dynamics model
        model = self._build_dynamics_model()
        
        # Build OCP
        ocp = AcadosOcp()
        ocp.model = model
        
        self.nx = model.x.rows()
        self.nu = model.u.rows()
        ocp.solver_options.N_horizon = self.N
        
        # External cost
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = self._build_cost_expression()
        
        # State constraints
        thrust_min = float(self._dyn_params["thrust_min"]) * 4.0
        thrust_max = float(self._dyn_params["thrust_max"]) * 4.0
        
        # [f, f_cmd, r_cmd, p_cmd, y_cmd]
        ocp.constraints.lbx = np.array([thrust_min, thrust_min, -1.57, -1.57, -1.57])
        ocp.constraints.ubx = np.array([thrust_max, thrust_max, 1.57, 1.57, 1.57])
        ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])
        
        # Input constraints
        # [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta_cmd]
        ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, 0.0])
        ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0, 1.8])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])
        
        # Initial state (will be overwritten)
        ocp.constraints.x0 = np.zeros(self.nx)
        
        # Parameters
        param_vec = self._encode_trajectory_params()
        ocp.parameter_values = param_vec
        
        # Solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tol = 1e-5
        ocp.solver_options.qp_solver_cond_N = self.N
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.qp_solver_iter_max = 20
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.tf = self.T
        
        # Create solver
        self.solver = AcadosOcpSolver(ocp, json_file="mpcc_racing.json", verbose=False)
        self.ocp = ocp
    
    def _build_dynamics_model(self) -> AcadosModel:
        """Build the quadrotor dynamics model."""
        model_name = "mpcc_drone_racing"
        
        # Dynamic parameters
        mass = self._mass
        gravity = self._gravity

        # Rate model parameters derived from drone-models
        k = np.array(self._dyn_params["rpy_coef"], dtype=float)          # [k_roll, k_pitch, k_yaw]
        d = np.array(self._dyn_params["rpy_rates_coef"], dtype=float)    # [d_roll, d_pitch, d_yaw]
        b = np.array(self._dyn_params["cmd_rpy_coef"], dtype=float)      # [b_roll, b_pitch, b_yaw]

        eps = 1e-9  # avoid divide-by-zero
        a = -k / (d + eps)
        beta = -b / (d + eps)

        params_roll_rate  = [float(a[0]), float(beta[0])]
        params_pitch_rate = [float(a[1]), float(beta[1])]
        params_yaw_rate   = [float(a[2]), float(beta[2])]
        
        # State variables
        self.px = MX.sym("px")
        self.py = MX.sym("py")
        self.pz = MX.sym("pz")
        self.vx = MX.sym("vx")
        self.vy = MX.sym("vy")
        self.vz = MX.sym("vz")
        self.roll = MX.sym("roll")
        self.pitch = MX.sym("pitch")
        self.yaw = MX.sym("yaw")
        self.f_collective = MX.sym("f_collective")
        self.f_cmd = MX.sym("f_cmd")
        self.r_cmd = MX.sym("r_cmd")
        self.p_cmd = MX.sym("p_cmd")
        self.y_cmd = MX.sym("y_cmd")
        self.theta = MX.sym("theta")  # Progress along path
        
        # Input variables
        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")
        self.v_theta_cmd = MX.sym("v_theta_cmd")  # Progress speed
        
        # State and input vectors
        states = vertcat(
            self.px, self.py, self.pz,
            self.vx, self.vy, self.vz,
            self.roll, self.pitch, self.yaw,
            self.f_collective, self.f_cmd,
            self.r_cmd, self.p_cmd, self.y_cmd,
            self.theta
        )
        inputs = vertcat(
            self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd,
            self.v_theta_cmd
        )
        
        # Dynamics equations
        thrust = self.f_collective
        inv_mass = 1.0 / mass
        
        # Acceleration from thrust
        ax = inv_mass * thrust * (
            cos(self.roll) * sin(self.pitch) * cos(self.yaw)
            + sin(self.roll) * sin(self.yaw)
        )
        ay = inv_mass * thrust * (
            cos(self.roll) * sin(self.pitch) * sin(self.yaw)
            - sin(self.roll) * cos(self.yaw)
        )
        az = inv_mass * thrust * cos(self.roll) * cos(self.pitch) - gravity
        
        # Continuous dynamics
        f_dyn = vertcat(
            self.vx,
            self.vy,
            self.vz,
            ax,
            ay,
            az,
            params_roll_rate[0] * self.roll + params_roll_rate[1] * self.r_cmd,
            params_pitch_rate[0] * self.pitch + params_pitch_rate[1] * self.p_cmd,
            params_yaw_rate[0] * self.yaw + params_yaw_rate[1] * self.y_cmd,
            10.0 * (self.f_cmd - self.f_collective),
            self.df_cmd,
            self.dr_cmd,
            self.dp_cmd,
            self.dy_cmd,
            self.v_theta_cmd
        )
        
        # Parameters for trajectory (positions, tangents, cost weights)
        n_samples = int(self.model_traj_length / self.model_arc_step)
        self.pd_list = MX.sym("pd_list", 3 * n_samples)
        self.tp_list = MX.sym("tp_list", 3 * n_samples)
        self.qc_dyn = MX.sym("qc_dyn", n_samples)
        params = vertcat(self.pd_list, self.tp_list, self.qc_dyn)
        
        # Build model
        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f_dyn
        model.x = states
        model.u = inputs
        model.p = params
        
        return model
    
    def _piecewise_linear_interp(self, theta, theta_vec, flattened_points, dim: int = 3):
        """CasADi-friendly linear interpolation."""
        M = len(theta_vec)
        idx_float = (theta - theta_vec[0]) / (theta_vec[-1] - theta_vec[0]) * (M - 1)
        
        idx_low = floor(idx_float)
        idx_high = idx_low + 1
        alpha = idx_float - idx_low
        
        idx_low = if_else(idx_low < 0, 0, idx_low)
        idx_high = if_else(idx_high >= M, M - 1, idx_high)
        
        p_low = vertcat(*[flattened_points[dim * idx_low + i] for i in range(dim)])
        p_high = vertcat(*[flattened_points[dim * idx_high + i] for i in range(dim)])
        
        return (1.0 - alpha) * p_low + alpha * p_high
    
    def _encode_trajectory_params(self) -> np.ndarray:
        """Encode trajectory for MPCC cost function."""
        cfg = self.mpcc_cfg
        theta_samples = np.arange(0.0, self.model_traj_length, self.model_arc_step)
        
        # Sample positions and tangents
        pd_vals = self.arc_trajectory(theta_samples)
        tp_vals = self.arc_trajectory.derivative(1)(theta_samples)
        
        # Dynamic cost weights (higher near gates and obstacles)
        qc_dyn = np.zeros_like(theta_samples)
        
        # Gate proximity
        for gate_center in self._cached_gate_centers:
            d_gate = np.linalg.norm(pd_vals - gate_center, axis=-1)
            qc_gate = cfg.k_qc_gate * np.exp(-8.0 * d_gate**2)
            qc_dyn = np.maximum(qc_dyn, qc_gate)
        
        # Obstacle proximity
        for obst_center in self._cached_obstacles:
            d_obs_xy = np.linalg.norm(pd_vals[:, :2] - obst_center[:2], axis=-1)
            qc_obs = cfg.k_qc_obs * np.exp(-8.0 * d_obs_xy**2)
            qc_dyn = np.maximum(qc_dyn, qc_obs)
        
        return np.concatenate([pd_vals.reshape(-1), tp_vals.reshape(-1), qc_dyn])
    
    def _build_cost_expression(self):
        """Build MPCC stage cost expression."""
        cfg = self.mpcc_cfg
        
        position = vertcat(self.px, self.py, self.pz)
        attitude = vertcat(self.roll, self.pitch, self.yaw)
        control = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)
        
        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_step)
        
        # Interpolate trajectory at current theta
        pd_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)
        qc_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_dyn, dim=1)
        
        # Compute tracking errors
        tp_unit = tp_theta / (norm_2(tp_theta) + 1e-6)
        e_theta = position - pd_theta
        e_lag = dot(tp_unit, e_theta) * tp_unit  # Lag error (along path)
        e_contour = e_theta - e_lag              # Contour error (perpendicular)
        
        # Tracking cost (with dynamic weights near gates)
        Q_w = cfg.q_attitude * DM(np.eye(3))
        track_cost = (
            (cfg.q_lag + cfg.q_lag_peak * qc_theta) * dot(e_lag, e_lag)
            + (cfg.q_contour + cfg.q_contour_peak * qc_theta) * dot(e_contour, e_contour)
            + attitude.T @ Q_w @ attitude
        )
        
        # Control smoothness cost
        R_df = DM(np.diag([cfg.r_thrust, cfg.r_roll, cfg.r_pitch, cfg.r_yaw]))
        smooth_cost = control.T @ R_df @ control
        
        # Speed incentive (maximize progress, but slow near gates)
        speed_cost = -cfg.mu_speed * self.v_theta_cmd + cfg.w_speed_gate * qc_theta * (self.v_theta_cmd**2)
        
        return track_cost + smooth_cost + speed_cost
    
    # =========================================================================
    # Environment Change Detection
    # =========================================================================
    
    def _detect_environment_change(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        """Detect changes in gate/obstacle visited flags."""
        if self._last_gate_flags is None:
            self._last_gate_flags = np.array(obs.get("gates_visited", []), dtype=bool)
            self._last_obst_flags = np.array(obs.get("obstacles_visited", []), dtype=bool)
            return False
        
        curr_gates = np.array(obs.get("gates_visited", []), dtype=bool)
        curr_obst = np.array(obs.get("obstacles_visited", []), dtype=bool)
        
        if curr_gates.shape != self._last_gate_flags.shape:
            self._last_gate_flags = curr_gates
            return False
        if curr_obst.shape != self._last_obst_flags.shape:
            self._last_obst_flags = curr_obst
            return False
        
        gate_trigger = np.any((~self._last_gate_flags) & curr_gates)
        obst_trigger = np.any((~self._last_obst_flags) & curr_obst)
        
        # Update gate detection status and record real positions
        for i, is_visited in enumerate(curr_gates):
            if is_visited and not self._gate_detected_flags[i]:
                # Gate newly detected - record its real position
                self._gate_detected_flags[i] = True
                self._gate_real_positions[i] = obs['gates_pos'][i]
                print(f"[GATE DETECTED] Gate {i+1} at real position: "
                      f"[{obs['gates_pos'][i][0]:.3f}, {obs['gates_pos'][i][1]:.3f}, {obs['gates_pos'][i][2]:.3f}]")
        
        self._last_gate_flags = curr_gates.copy()
        self._last_obst_flags = curr_obst.copy()
        
        return bool(gate_trigger or obst_trigger)
    
    # =========================================================================
    # Safety Checks
    # =========================================================================
    
    def _check_position_bounds(self, pos: NDArray[np.floating]) -> bool:
        """Check if position is within safe bounds."""
        bounds = self.mpcc_cfg.pos_bounds
        for i, (low, high) in enumerate(bounds):
            if pos[i] < low or pos[i] > high:
                return False
        return True
    
    def _check_velocity_bounds(self, vel: NDArray[np.floating]) -> bool:
        """Check if velocity is within safe bounds."""
        speed = np.linalg.norm(vel)
        low, high = self.mpcc_cfg.vel_bounds
        return low < speed < high
    
    # =========================================================================
    # Main Control Loop
    # =========================================================================
    
    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None
    ) -> NDArray[np.floating]:
        """
        Compute control command using MPCC.
        
        Args:
            obs: Current observation.
            info: Optional additional info.
            
        Returns:
            Control command [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd].
        """
        self._current_pos = obs["pos"].copy()
        if not self._actual_path_points:
            self._actual_path_points.append(obs["pos"].copy())
        
        # Check for environment changes
        if self._detect_environment_change(obs):
            print(f"[MPCC] Environment change detected, replanning...")
            self._plan_trajectory(obs)
            # Re-project theta with local-window search to avoid branch jumps.
            try:
                self.last_theta = self._project_theta_local(obs["pos"], center_theta=self.last_theta)
            except Exception as e:
                print(f"[MPCC] Warning: could not project theta after replanning: {e}")

            # Reset warm start so optimizer does not drag old path states.
            if hasattr(self, "_x_warm"):
                del self._x_warm
            if hasattr(self, "_u_warm"):
                del self._u_warm
                
            # Update solver parameters
            param_vec = self._encode_trajectory_params()
            for k in range(self.N + 1):
                self.solver.set(k, "p", param_vec)

        # Always keep theta on the local branch and active stage.
        self.last_theta = self._project_theta_local(obs["pos"], center_theta=self.last_theta)
        self._apply_stage_progress_limit(obs)
        
        # Convert quaternion to Euler angles
        quat = obs["quat"]
        roll, pitch, yaw = Rotation.from_quat(quat).as_euler("xyz")
        
        # Build current state
        x_now = np.concatenate([
            obs["pos"],
            obs["vel"],
            np.array([roll, pitch, yaw]),
            np.array([self.last_f_collective, self.last_f_cmd]),
            self.last_rpy_cmd,
            np.array([self.last_theta])
        ])
        
        # Warm start
        if not hasattr(self, "_x_warm"):
            self._x_warm = [x_now.copy() for _ in range(self.N + 1)]
            self._u_warm = [np.zeros(self.nu) for _ in range(self.N)]
        else:
            self._x_warm = self._x_warm[1:] + [self._x_warm[-1]]
            self._u_warm = self._u_warm[1:] + [self._u_warm[-1]]
        
        # Set initial guess
        for i in range(self.N):
            self.solver.set(i, "x", self._x_warm[i])
            self.solver.set(i, "u", self._u_warm[i])
        self.solver.set(self.N, "x", self._x_warm[self.N])
        
        # Fix initial state
        self.solver.set(0, "lbx", x_now)
        self.solver.set(0, "ubx", x_now)
        
        # Check termination conditions
        if self.last_theta >= float(self.arc_trajectory.x[-1]):
            self.finished = True
            print("[MPCC] Finished: reached end of path")
        
        if not self._check_position_bounds(obs["pos"]):
            self.finished = True
            print("[MPCC] Finished: position out of bounds")
        
        if not self._check_velocity_bounds(obs["vel"]):
            self.finished = True
            print("[MPCC] Finished: velocity out of bounds")
        
        # Solve MPC
        status = self.solver.solve()
        if status != 0:
            print(f"[MPCC] Solver returned status {status}")
        
        # Extract solution
        self._x_warm = [self.solver.get(i, "x") for i in range(self.N + 1)]
        self._u_warm = [self.solver.get(i, "u") for i in range(self.N)]
        
        x_next = self.solver.get(1, "x")
        
        # Update last commands
        self.last_f_collective = float(x_next[9])
        self.last_f_cmd = float(x_next[10])
        self.last_rpy_cmd = np.array(x_next[11:14])
        self.last_theta = min(float(x_next[14]), float(self._stage_theta_end))
        
        # Build output command [roll, pitch, yaw, thrust]
        cmd = np.array([
            self.last_rpy_cmd[0],
            self.last_rpy_cmd[1],
            self.last_rpy_cmd[2],
            self.last_f_cmd
        ], dtype=np.float32)
        
        # Periodic logging
        if self._step_count % self.mpcc_cfg.log_interval == 0:
            print(f"[MPCC] T={self._step_count / self._ctrl_freq:.2f}s, "
                  f"theta={self.last_theta:.2f}/{self.arc_trajectory.x[-1]:.2f}, "
                  f"cmd=[{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}, {cmd[3]:.2f}]")
        
        self._step_count += 1
        return cmd
    
    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict
    ) -> bool:
        """Called after each step."""
        self._actual_path_points.append(obs["pos"].copy())
        if len(self._actual_path_points) > 5000:
            self._actual_path_points = self._actual_path_points[-5000:]
        return self.finished
    
    def episode_callback(self):
        """Called at episode reset."""
        print("[MPCC] Episode reset")
        self._step_count = 0
        self.finished = False
        
        # Clear cached states
        for attr in ["_last_gate_flags", "_last_obst_flags", "_x_warm", "_u_warm"]:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Reset control states
        self.last_theta = 0.0
        self.last_f_collective = self.hover_thrust
        self.last_f_cmd = self.hover_thrust
        self.last_rpy_cmd = np.zeros(3)
        self._actual_path_points = []
    
    # =========================================================================
    # Debug Methods
    # =========================================================================
    
    def get_debug_lines(self):
        """Return line segments for visualization."""
        debug_lines = []
        
        # Planned complete path (green)
        if self._planned_path_points.shape[0] >= 1:
            try:
                full_path = self._downsample_polyline(self._planned_path_points, max_points=400)
                if full_path.shape[0] == 1:
                    full_path = np.vstack([full_path, full_path])
                debug_lines.append(
                    (full_path, np.array([0.0, 1.0, 0.0, 0.9]), 2.0, 2.0)
                )
            except Exception:
                pass
        
        # Real flown path (red)
        if len(self._actual_path_points) >= 2:
            try:
                flown_path = np.asarray(self._actual_path_points, dtype=float)
                flown_path = self._downsample_polyline(flown_path, max_points=500)
                debug_lines.append(
                    (flown_path, np.array([1.0, 0.0, 0.0, 0.95]), 2.5, 2.5)
                )
            except Exception:
                pass

        # Line to current reference
        if hasattr(self, "last_theta") and hasattr(self, "arc_trajectory"):
            try:
                target = self.arc_trajectory(self.last_theta)
                segment = np.stack([self._current_pos, target])
                debug_lines.append(
                    (segment, np.array([0.0, 0.0, 1.0, 1.0]), 1.0, 1.0)
                )
            except Exception:
                pass
        
        return debug_lines

    def render_callback(self, sim: Sim):
        """Render debug lines in the simulator."""
        for points, rgba, min_size, max_size in self.get_debug_lines():
            draw_line(sim, points, rgba=tuple(rgba), start_size=min_size, end_size=max_size)
    
    def get_trajectory(self) -> CubicSpline:
        """Get the current trajectory spline."""
        return self.trajectory
    
    def get_arc_trajectory(self) -> CubicSpline:
        """Get the arc-length parameterized trajectory."""
        return self.arc_trajectory
    
    def get_progress(self) -> float:
        """Get current progress ratio (0-1)."""
        if hasattr(self, "arc_trajectory"):
            return self.last_theta / self.arc_trajectory.x[-1]
        return 0.0
