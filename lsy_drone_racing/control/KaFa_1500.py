"""Gate-centric full-state controller for the drone racing environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.kafa1500.commands import StateActionBuilder
from lsy_drone_racing.control.kafa1500.navigation import GateNavigator
from lsy_drone_racing.control.kafa1500.settings import ActionSettings, PlannerSettings
from lsy_drone_racing.control.kafa1500.types import GatePlan, KaFa1500State, Observation, Vec3

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class KaFa1500(Controller):
    """State-machine controller with local gate planning and obstacle routing."""

    def __init__(self, obs: Observation, info: dict, config: dict):
        """Initialize the controller state and helper modules."""
        super().__init__(obs, info, config)

        self._planner_settings = PlannerSettings()
        self._action_settings = ActionSettings()
        self._navigator = GateNavigator(self._planner_settings)

        self._start_pos = obs["pos"].astype(np.float32).copy()
        self._takeoff_height = float(max(0.8, self._start_pos[2] + 0.8))
        self._takeoff_yaw = StateActionBuilder.yaw_from_quat(obs["quat"])
        self._action_builder = StateActionBuilder(self._action_settings, self._takeoff_yaw)

        self._state = KaFa1500State.TAKEOFF
        self._tick = 0
        self._finished = False
        self._scan_started_at = 0
        self._last_target_gate = int(obs["target_gate"])
        self._last_command_state = self._state

        self._plan: GatePlan | None = None
        self._current_target = self._start_pos.copy()
        self._reference_speed = 0.0

    def compute_control(
        self,
        obs: Observation,
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        """Compute the next desired full-state target."""
        target_gate = int(obs["target_gate"])
        gate_changed = target_gate != self._last_target_gate

        if target_gate < 0:
            self._state = KaFa1500State.FINISH
        elif gate_changed and self._state not in (KaFa1500State.TAKEOFF, KaFa1500State.PASS_GATE):
            self._begin_scan()

        target_pos, yaw_dir, speed_limit, ref_vel, ref_acc = self._state_target(obs)
        state_changed = self._state != self._last_command_state

        self._last_target_gate = target_gate
        self._current_target = target_pos.copy()
        action = self._action_builder.build(
            obs=obs,
            target_pos=target_pos,
            yaw_dir=yaw_dir,
            speed_limit=speed_limit,
            state=self._state,
            state_changed=state_changed,
            ref_vel=ref_vel,
            ref_acc=ref_acc,
        )
        self._last_command_state = self._state
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the internal clock and report whether the run has finished."""
        self._tick += 1
        self._finished = bool(self._finished or terminated or truncated or obs["target_gate"] == -1)
        return self._finished

    def episode_callback(self):
        """Reset controller state between episodes."""
        self._state = KaFa1500State.TAKEOFF
        self._tick = 0
        self._finished = False
        self._scan_started_at = 0
        self._last_target_gate = 0
        self._last_command_state = self._state

        self._plan = None
        self._current_target = self._start_pos.copy()
        self._reference_speed = 0.0
        self._action_builder.reset()

    def render_callback(self, sim: Sim):
        """Visualize the current setpoint and the planned route."""
        draw_points(sim, self._current_target.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.03)
        if self._plan is not None:
            draw_line(sim, self._plan.route_line, rgba=(0.0, 1.0, 0.0, 1.0))

    def _state_target(self, obs: Observation) -> tuple[Vec3, Vec3, float, Vec3, Vec3]:
        """Select a target point and reference dynamics for the active high-level state."""
        pos = obs["pos"].astype(np.float32)
        target_gate = int(obs["target_gate"])

        if self._state == KaFa1500State.TAKEOFF:
            return self._takeoff_target(pos, target_gate)

        if self._state == KaFa1500State.SCAN:
            return self._scan_target(obs, pos, target_gate)

        if self._state == KaFa1500State.FINISH or target_gate < 0:
            return self._hold_target(pos, self._action_builder.heading_vector())

        if self._plan is None or (
            target_gate != self._plan.gate_idx and self._state != KaFa1500State.PASS_GATE
        ):
            self._begin_scan()
            return self._hold_target(pos, self._takeoff_heading())

        if self._state == KaFa1500State.APPROACH:
            return self._approach_target(pos)

        if self._state == KaFa1500State.PASS_GATE:
            return self._pass_gate_target(pos, target_gate)

        raise ValueError(f"Invalid state: {self._state}")

    def _takeoff_target(self, pos: Vec3, target_gate: int) -> tuple[Vec3, Vec3, float, Vec3, Vec3]:
        """Return the takeoff setpoint and advance once the drone is high enough."""
        target = np.array(
            [self._start_pos[0], self._start_pos[1], self._takeoff_height],
            dtype=np.float32,
        )
        if np.linalg.norm(target - pos) < self._planner_settings.takeoff_tol:
            if target_gate >= 0:
                self._begin_scan()
            else:
                self._state = KaFa1500State.FINISH
        zero = np.zeros(3, dtype=np.float32)
        return target, self._takeoff_heading(), self._action_settings.takeoff_speed, zero, zero

    def _scan_target(
        self,
        obs: Observation,
        pos: Vec3,
        target_gate: int,
    ) -> tuple[Vec3, Vec3, float, Vec3, Vec3]:
        """Plan from the latest observed track data while commanding a short brake."""
        if target_gate < 0:
            self._state = KaFa1500State.FINISH
            return self._hold_target(pos, self._action_builder.heading_vector())

        self._plan = self._navigator.plan_gate(
            obs,
            target_gate,
            start_pos=pos,
            start_vel=obs["vel"].astype(np.float32),
        )
        scan_steps = self._tick - self._scan_started_at
        horizontal_speed = float(np.linalg.norm(obs["vel"][:2]))
        min_hold_elapsed = scan_steps >= self._planner_settings.scan_hold_steps
        speed_settled = horizontal_speed <= self._planner_settings.scan_release_speed
        max_hold_elapsed = scan_steps >= self._planner_settings.max_scan_hold_steps

        if min_hold_elapsed and (speed_settled or max_hold_elapsed):
            self._state = KaFa1500State.APPROACH
            return self._approach_target(pos)
        return self._hold_target(pos, self._takeoff_heading())

    def _approach_target(self, pos: Vec3) -> tuple[Vec3, Vec3, float, Vec3, Vec3]:
        """Follow the planned path until the final gate pass segment."""
        if self._plan is None:
            return self._hold_target(pos, self._takeoff_heading())

        route_speed = self._smooth_speed(self._action_settings.route_speed)
        path_target = self._navigator.follow_path(pos, self._plan, route_speed)
        signed_progress = float(np.dot(pos - self._plan.gate_pos, self._plan.gate_x))
        if signed_progress > -self._planner_settings.approach_tol:
            self._state = KaFa1500State.PASS_GATE
            pass_speed = self._smooth_speed(self._action_settings.pass_speed)
            pass_target = self._navigator.follow_path(pos, self._plan, pass_speed)
            return (
                pass_target.target,
                self._plan.gate_x,
                pass_speed,
                pass_target.ref_vel,
                pass_target.ref_acc,
            )

        final_corridor = path_target.remaining <= self._planner_settings.final_corridor_distance
        yaw_dir = self._plan.gate_x if final_corridor else path_target.yaw_dir
        speed = (
            self._action_settings.approach_speed
            if final_corridor
            else self._action_settings.route_speed
        )
        speed = self._smooth_speed(speed)
        path_target = self._navigator.follow_path(pos, self._plan, speed)
        return (
            path_target.target,
            yaw_dir.astype(np.float32),
            speed,
            path_target.ref_vel,
            path_target.ref_acc,
        )

    def _pass_gate_target(self, pos: Vec3, target_gate: int) -> tuple[Vec3, Vec3, float, Vec3, Vec3]:
        """Keep driving through the current gate until the environment registers it."""
        if self._plan is None:
            return self._hold_target(pos, self._takeoff_heading())
        if target_gate < 0:
            self._state = KaFa1500State.FINISH
            return self._hold_target(pos, self._action_builder.heading_vector())
        if target_gate != self._plan.gate_idx:
            self._begin_scan()
            return self._hold_target(pos, self._takeoff_heading())

        pass_speed = self._smooth_speed(self._action_settings.pass_speed)
        path_target = self._navigator.follow_path(pos, self._plan, pass_speed)
        signed_progress = float(np.dot(pos - self._plan.gate_pos, self._plan.gate_x))
        if signed_progress > self._planner_settings.max_pass_overshoot:
            self._begin_scan()
            return self._hold_target(pos, self._takeoff_heading())
        return (
            path_target.target,
            self._plan.gate_x,
            pass_speed,
            path_target.ref_vel,
            path_target.ref_acc,
        )

    def _hold_target(self, pos: Vec3, yaw_dir: Vec3) -> tuple[Vec3, Vec3, float, Vec3, Vec3]:
        """Hold the current XY position while preserving safe flight height."""
        target = pos.copy()
        target[2] = max(target[2], self._takeoff_height)
        self._reference_speed *= 0.7
        zero = np.zeros(3, dtype=np.float32)
        return target, yaw_dir, 0.0, zero, zero

    def _begin_scan(self) -> None:
        """Enter scan mode and invalidate the previous gate plan."""
        self._state = KaFa1500State.SCAN
        self._scan_started_at = self._tick
        self._plan = None

    def _takeoff_heading(self) -> Vec3:
        """Return the fixed heading used during takeoff and scan."""
        return np.array(
            [np.cos(self._takeoff_yaw), np.sin(self._takeoff_yaw), 0.0],
            dtype=np.float32,
        )

    def _smooth_speed(self, target_speed: float) -> float:
        """Low-pass filter reference speed to avoid discontinuous replanning handoffs."""
        gain = self._action_settings.speed_transition_gain
        self._reference_speed = (1.0 - gain) * self._reference_speed + gain * target_speed
        return float(max(0.0, self._reference_speed))
