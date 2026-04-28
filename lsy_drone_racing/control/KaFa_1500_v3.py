from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.utils.planner import Plan, PlannerConfig, build_plan
from lsy_drone_racing.control.utils.racing_line import RacingLineConfig, build_racing_line_plan

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Unique suffix for acados generated artifacts so parallel benchmark workers
# (each with a distinct ``LSY_FAST_WORKER_ID``) do not clobber each other's
# generated C code / JSON.
_WORKER_SUFFIX = (
    f"_w{os.environ['LSY_FAST_WORKER_ID']}" if os.environ.get("LSY_FAST_WORKER_ID") else ""
)
_MODEL_NAME = f"gate_aware_fast_v3_s55_t757{_WORKER_SUFFIX}"

_DIAG_PATH = Path("/tmp/lsy_diagnostics.csv")


def _build_ocp(
    horizon_time: float,
    horizon_steps: int,
    parameters: dict,
    n_obstacles: int,
    r_safe: float,
    w_obs: float,
    n_wings: int = 4,
    r_wing: float = 0.13,
    w_wing: float = 80000.0,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Build an acados OCP with parametric soft obstacle + current-gate wing constraints.

    Parameters (``model.p``) layout:
      [0 : 2*n_obstacles]      — obstacles xy (2D), constrained in xy
      [2*n_obstacles : 2*n_obstacles + 3*n_wings] — wing centers xyz (3D)
    """
    state_derivative, state, control, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    parameter_count = 2 * n_obstacles + 3 * n_wings
    obstacle_wing_params = ca.MX.sym("obs_p", parameter_count)
    model = AcadosModel()
    model.name = _MODEL_NAME
    model.f_expl_expr = state_derivative
    model.f_impl_expr = None
    model.x = state
    model.u = control
    model.p = obstacle_wing_params

    ocp = AcadosOcp()
    ocp.model = model

    state_dim = state.rows()
    control_dim = control.rows()
    reference_dim = state_dim + control_dim
    terminal_reference_dim = state_dim

    ocp.solver_options.N_horizon = horizon_steps

    # Tuned for cf21B_500 (m≈43g, T/W≈1.9). xy weights 75 (vs base 50) for
    # tighter lateral tracking through the 0.4 m gate openings; vel weights
    # 12 (vs 10) for smoother approaches.
    state_weights = np.diag([75.0, 75.0, 400.0, 1.0, 1.0, 1.0, 12.0, 12.0, 12.0, 5.0, 5.0, 5.0])
    control_weights = np.diag([1.0, 1.0, 1.0, 50.0])
    terminal_weights = state_weights.copy()

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(state_weights, control_weights)
    ocp.cost.W_e = terminal_weights

    state_reference_matrix = np.zeros((reference_dim, state_dim))
    state_reference_matrix[0:state_dim, 0:state_dim] = np.eye(state_dim)
    ocp.cost.Vx = state_reference_matrix
    control_reference_matrix = np.zeros((reference_dim, control_dim))
    control_reference_matrix[state_dim : state_dim + control_dim, :] = np.eye(control_dim)
    ocp.cost.Vu = control_reference_matrix
    terminal_state_reference_matrix = np.zeros((terminal_reference_dim, state_dim))
    terminal_state_reference_matrix[0:state_dim, 0:state_dim] = np.eye(state_dim)
    ocp.cost.Vx_e = terminal_state_reference_matrix
    ocp.cost.yref = np.zeros((reference_dim,))
    ocp.cost.yref_e = np.zeros((terminal_reference_dim,))

    # Attitude state bounds on roll/pitch kept generous (±1.20 rad ≈ 69°) so
    # that the MPC's QP stays feasible even when the drone physically tilts
    # past the "preferred" range during aggressive gate transitions. The cost
    # function (Q diag=1.0 on rpy) still discourages large tilts, but we no
    # longer make the OCP infeasible when x0's measured rpy exceeds the bound.
    # Command (input) bounds remain tighter at ±0.80 rad since they directly
    # drive the attitude controller; the drone's actual roll can overshoot
    # commanded via body dynamics under large angular rates.
    # Yaw bound widened to ±1.5 rad: at ±0.5 the OCP went infeasible during
    # fast gate-2/3 transitions, causing stale-control ground strikes.
    ocp.constraints.lbx = np.array([-1.20, -1.20, -1.5])
    ocp.constraints.ubx = np.array([1.20, 1.20, 1.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])
    ocp.constraints.lbu = np.array([-0.80, -0.80, -1.0, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.80, 0.80, 1.0, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.zeros((state_dim,))

    # Soft constraints: squared 2D distance to each obstacle >= r_safe**2, plus
    # squared 3D distance to each wing of the current target gate >= r_wing**2.
    drone_xy = state[0:2]
    drone_xyz = state[0:3]
    squared_distance_terms = []
    for obstacle_index in range(n_obstacles):
        obstacle_x = obstacle_wing_params[2 * obstacle_index]
        obstacle_y = obstacle_wing_params[2 * obstacle_index + 1]
        dx = drone_xy[0] - obstacle_x
        dy = drone_xy[1] - obstacle_y
        squared_distance_terms.append(dx * dx + dy * dy)
    wing_offset = 2 * n_obstacles
    for wing_index in range(n_wings):
        wing_x = obstacle_wing_params[wing_offset + 3 * wing_index]
        wing_y = obstacle_wing_params[wing_offset + 3 * wing_index + 1]
        wing_z = obstacle_wing_params[wing_offset + 3 * wing_index + 2]
        dx = drone_xyz[0] - wing_x
        dy = drone_xyz[1] - wing_y
        dz = drone_xyz[2] - wing_z
        squared_distance_terms.append(dx * dx + dy * dy + dz * dz)
    soft_constraint_count = n_obstacles + n_wings
    model.con_h_expr = ca.vertcat(*squared_distance_terms)
    ocp.constraints.lh = np.concatenate(
        [np.full(n_obstacles, r_safe**2), np.full(n_wings, r_wing**2)]
    )
    ocp.constraints.uh = np.full(soft_constraint_count, 1e6)
    ocp.constraints.idxsh = np.arange(soft_constraint_count)
    ocp.cost.zl = np.zeros(soft_constraint_count)
    ocp.cost.zu = np.zeros(soft_constraint_count)
    ocp.cost.Zl = np.concatenate([np.full(n_obstacles, w_obs), np.full(n_wings, w_wing)])
    ocp.cost.Zu = np.zeros(soft_constraint_count)

    ocp.parameter_values = np.zeros(parameter_count)

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.qp_solver_cond_N = horizon_steps
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 40
    ocp.solver_options.nlp_solver_max_iter = 10
    ocp.solver_options.tf = horizon_time

    solver = AcadosOcpSolver(ocp, json_file=f"c_generated_code/{_MODEL_NAME}.json", verbose=False)
    return solver, ocp


class GateAwareFastV3S55T757(Controller):
    """Attitude-mode MPC with parametric obstacle soft constraints."""

    _run_counter = 0

    N = 35
    PLAN_PAD = 200
    N_OBSTACLES = 4
    N_WINGS = 4
    R_SAFE = 0.20
    R_WING = 0.13
    W_OBS = 150000.0
    W_WING = 80000.0
    WING_OFFSET = 0.28  # distance along gate y/z axis to wing midpoint
    USE_RACING_LINE = False
    PLANNER = PlannerConfig(
        d_pre=0.28, d_post=0.18, v_cruise=2.60, v_cruise_inter=4.00, t_min_seg=0.24, r_obs=0.22
    )
    RACING_LINE = RacingLineConfig(
        v_cruise=1.8, t_min_seg=0.15, max_accel=9.0, max_vel=4.0, r_obs=0.22
    )

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict) -> None:
        """Build MPC + initial plan."""
        super().__init__(obs, info, config)
        self._dt = 1.0 / config.env.freq
        self._horizon_time = self.N * self._dt

        self._drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = _build_ocp(
            self._horizon_time,
            self.N,
            self._drone_params,
            self.N_OBSTACLES,
            self.R_SAFE,
            self.W_OBS,
            n_wings=self.N_WINGS,
            r_wing=self.R_WING,
            w_wing=self.W_WING,
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._finished = False
        self._plan_spline_ticks = 0
        self._pos_samples = np.zeros((0, 3))
        self._vel_samples = np.zeros((0, 3))
        self._replan(obs, start_vel=np.zeros(3), target_gate=0)

        self._prev_gates_visited = np.asarray(obs["gates_visited"]).copy()
        self._prev_obstacles_visited = np.asarray(obs["obstacles_visited"]).copy()

        GateAwareFastV3S55T757._run_counter += 1
        self._run_idx = GateAwareFastV3S55T757._run_counter
        self._terminal_state: dict | None = None
        self._last_live_state: dict | None = None
        self._flight_tick = 0  # never reset by _replan
        if self._run_idx == 1 and _DIAG_PATH.exists():
            _DIAG_PATH.unlink()

    # ---- planning ---------------------------------------------------------

    def _replan(
        self,
        obs: dict[str, NDArray[np.floating]],
        start_vel: NDArray[np.floating],
        target_gate: int,
    ) -> None:
        if self.USE_RACING_LINE:
            plan: Plan = build_racing_line_plan(
                start_pos=np.asarray(obs["pos"], dtype=np.float64),
                start_vel=np.asarray(start_vel, dtype=np.float64),
                gates_pos=obs["gates_pos"],
                gates_quat=obs["gates_quat"],
                obstacles_pos=obs["obstacles_pos"],
                target_gate=target_gate,
                cfg=self.RACING_LINE,
            )
        else:
            plan = build_plan(
                start_pos=np.asarray(obs["pos"], dtype=np.float64),
                start_vel=np.asarray(start_vel, dtype=np.float64),
                gates_pos=obs["gates_pos"],
                gates_quat=obs["gates_quat"],
                obstacles_pos=obs["obstacles_pos"],
                target_gate=target_gate,
                cfg=self.PLANNER,
            )
        sample_count = max(int(np.ceil(plan.t_total / self._dt)), 2)
        sample_times = np.arange(sample_count) * self._dt
        sample_times = np.clip(sample_times, 0.0, plan.t_total)
        position_samples = plan.pos_spline(sample_times)
        velocity_samples = plan.vel_spline(sample_times)
        padded_positions = np.tile(position_samples[-1], (self.PLAN_PAD, 1))
        padded_velocities = np.zeros((self.PLAN_PAD, 3))
        self._pos_samples = np.vstack([position_samples, padded_positions])
        self._vel_samples = np.vstack([velocity_samples, padded_velocities])
        self._plan_spline_ticks = sample_count
        self._tick = 0

    def _current_gate_wings(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Return (N_WINGS, 3) world-frame midpoints of the current target gate's frame wings.

        Frame wings are left/right along the gate y-axis and top/bottom along world z.
        Parked far away when no live gate exists (target_gate == -1, or out of range).
        """
        target_gate = int(obs["target_gate"])
        if target_gate < 0 or target_gate >= len(obs["gates_pos"]):
            return np.full((self.N_WINGS, 3), 100.0)
        gate_position = np.asarray(obs["gates_pos"][target_gate], dtype=np.float64)
        gate_lateral_axis = R.from_quat(obs["gates_quat"][target_gate]).as_matrix()[:, 1]
        left_wing = gate_position + self.WING_OFFSET * gate_lateral_axis
        right_wing = gate_position - self.WING_OFFSET * gate_lateral_axis
        top_wing = gate_position + self.WING_OFFSET * np.array([0.0, 0.0, 1.0])
        bottom_wing = gate_position - self.WING_OFFSET * np.array([0.0, 0.0, 1.0])
        return np.stack([left_wing, right_wing, top_wing, bottom_wing])

    # ---- controller API ---------------------------------------------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the MPC's first-step attitude command for the current observation."""
        if self._tick >= self._plan_spline_ticks + self.PLAN_PAD - self.N - 1:
            self._finished = True
        plan_index = min(self._tick, self._plan_spline_ticks + self.PLAN_PAD - self.N - 2)

        attitude_rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        rpy_rates = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        measured_state = np.concatenate((obs["pos"], attitude_rpy, obs["vel"], rpy_rates))
        self._acados_ocp_solver.set(0, "lbx", measured_state)
        self._acados_ocp_solver.set(0, "ubx", measured_state)

        # Build parameter vector: [obstacles_xy (flat), wings_xyz (flat)].
        obstacle_xy_params = np.asarray(obs["obstacles_pos"])[: self.N_OBSTACLES, :2].flatten()
        if obstacle_xy_params.size < 2 * self.N_OBSTACLES:
            obstacle_xy_params = np.concatenate(
                [obstacle_xy_params, np.full(2 * self.N_OBSTACLES - obstacle_xy_params.size, 1e6)]
            )
        gate_wing_params = self._current_gate_wings(obs).flatten()
        parameter_vector = np.concatenate([obstacle_xy_params, gate_wing_params])
        for stage_index in range(self.N + 1):
            self._acados_ocp_solver.set(stage_index, "p", parameter_vector)

        hover_thrust = self._drone_params["mass"] * -self._drone_params["gravity_vec"][-1]
        stage_reference = np.zeros((self.N, self._ny))
        stage_reference[:, 0:3] = self._pos_samples[plan_index : plan_index + self.N]
        stage_reference[:, 6:9] = self._vel_samples[plan_index : plan_index + self.N]
        stage_reference[:, 15] = hover_thrust
        for stage_index in range(self.N):
            self._acados_ocp_solver.set(stage_index, "yref", stage_reference[stage_index])

        terminal_reference = np.zeros((self._ny_e,))
        terminal_reference[0:3] = self._pos_samples[plan_index + self.N]
        terminal_reference[6:9] = self._vel_samples[plan_index + self.N]
        self._acados_ocp_solver.set(self.N, "y_ref", terminal_reference)

        self._acados_ocp_solver.solve()
        first_control = self._acados_ocp_solver.get(0, "u")
        return np.asarray(first_control, dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance tick counters, snapshot terminal state, and trigger replans on sense events."""
        self._tick += 1
        self._flight_tick += 1
        target_gate = int(obs["target_gate"])

        current_position = np.asarray(obs["pos"]).copy()
        disabled_position = bool(np.all(np.isclose(current_position, -1.0, atol=1e-3)))
        if not disabled_position:
            self._last_live_state = {
                "tick": self._flight_tick,
                "pos": current_position,
                "vel": np.asarray(obs["vel"]).copy(),
                "target_gate": target_gate,
                "gates_pos": np.asarray(obs["gates_pos"]).copy(),
                "gates_quat": np.asarray(obs["gates_quat"]).copy(),
                "gates_visited": np.asarray(obs["gates_visited"]).copy(),
                "obstacles_pos": np.asarray(obs["obstacles_pos"]).copy(),
            }

        if terminated or truncated:
            last_live_state = self._last_live_state or {}
            self._terminal_state = {
                **last_live_state,
                "final_target_gate": target_gate,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }

        if target_gate == -1:
            self._finished = True
            return self._finished
        gates_visited = np.asarray(obs["gates_visited"])
        obstacles_visited = np.asarray(obs["obstacles_visited"])
        detected_new_gate = bool((gates_visited & ~self._prev_gates_visited).any())
        detected_new_obstacle = bool((obstacles_visited & ~self._prev_obstacles_visited).any())
        if detected_new_gate or detected_new_obstacle:
            self._replan(obs, start_vel=np.asarray(obs["vel"]), target_gate=target_gate)
        self._prev_gates_visited = gates_visited.copy()
        self._prev_obstacles_visited = obstacles_visited.copy()
        return self._finished

    def episode_callback(self) -> None:
        """Write a diagnostic row at episode end and reset per-episode state."""
        if self._terminal_state is not None:
            self._write_diagnostic(self._terminal_state)
        self._tick = 0
        self._flight_tick = 0
        self._terminal_state = None
        self._last_live_state = None

    def _write_diagnostic(self, terminal_snapshot: dict) -> None:
        position = terminal_snapshot.get("pos", np.zeros(3))
        velocity = terminal_snapshot.get("vel", np.zeros(3))
        target_gate = terminal_snapshot.get(
            "final_target_gate", terminal_snapshot.get("target_gate", -2)
        )
        flight_time = terminal_snapshot.get("tick", 0) * self._dt
        success = target_gate == -1
        outcome = (
            "success" if success else ("timeout" if terminal_snapshot.get("truncated") else "crash")
        )

        gate_positions = terminal_snapshot["gates_pos"]
        obstacle_positions = terminal_snapshot["obstacles_pos"]
        gate_distances = np.linalg.norm(gate_positions - position[None, :], axis=1)
        obstacle_distances_2d = np.linalg.norm(
            obstacle_positions[:, :2] - position[None, :2], axis=1
        )
        nearest_obstacle_index = int(np.argmin(obstacle_distances_2d))
        nearest_obstacle_distance = float(obstacle_distances_2d[nearest_obstacle_index])
        nearest_gate_index = int(np.argmin(gate_distances))
        nearest_gate_distance = float(gate_distances[nearest_gate_index])

        if 0 <= target_gate < len(gate_positions):
            gate_rotation = R.from_quat(terminal_snapshot["gates_quat"][target_gate]).as_matrix()
            relative_position = position - gate_positions[target_gate]
            local_x = float(gate_rotation[:, 0] @ relative_position)
            local_y = float(gate_rotation[:, 1] @ relative_position)
            local_z = float(relative_position[2])
        else:
            local_x = local_y = local_z = 0.0

        header = not _DIAG_PATH.exists()
        with open(_DIAG_PATH, "a", newline="") as diagnostic_file:
            writer = csv.writer(diagnostic_file)
            if header:
                writer.writerow(
                    [
                        "run",
                        "outcome",
                        "t_flight",
                        "target_gate",
                        "pos_x",
                        "pos_y",
                        "pos_z",
                        "vx",
                        "vy",
                        "vz",
                        "speed",
                        "near_obs_idx",
                        "near_obs_xy_dist",
                        "near_gate_idx",
                        "near_gate_3d_dist",
                        "tgt_local_x",
                        "tgt_local_y",
                        "tgt_local_z",
                        "gate0_dist",
                        "gate1_dist",
                        "gate2_dist",
                        "gate3_dist",
                        "obs0_xy",
                        "obs1_xy",
                        "obs2_xy",
                        "obs3_xy",
                    ]
                )
            writer.writerow(
                [
                    self._run_idx,
                    outcome,
                    f"{flight_time:.3f}",
                    target_gate,
                    f"{position[0]:.3f}",
                    f"{position[1]:.3f}",
                    f"{position[2]:.3f}",
                    f"{velocity[0]:.3f}",
                    f"{velocity[1]:.3f}",
                    f"{velocity[2]:.3f}",
                    f"{float(np.linalg.norm(velocity)):.3f}",
                    nearest_obstacle_index,
                    f"{nearest_obstacle_distance:.3f}",
                    nearest_gate_index,
                    f"{nearest_gate_distance:.3f}",
                    f"{local_x:.3f}",
                    f"{local_y:.3f}",
                    f"{local_z:.3f}",
                    *[f"{distance:.3f}" for distance in gate_distances],
                    *[f"{distance:.3f}" for distance in obstacle_distances_2d],
                ]
            )
        self._tick = 0
