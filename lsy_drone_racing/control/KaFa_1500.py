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
        elif gate_changed and self._state != KaFa1500State.TAKEOFF:
            self._begin_scan()

        target_pos, yaw_dir, speed_limit = self._state_target(obs)
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
        self._action_builder.reset()

    def render_callback(self, sim: Sim):
        """Visualize the current setpoint and the planned route."""

        draw_points(sim, self._current_target.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.03)
        if self._plan is not None:
            draw_line(sim, self._plan.route_line, rgba=(0.0, 1.0, 0.0, 1.0))

    def _state_target(self, obs: Observation) -> tuple[Vec3, Vec3, float]:
        """Select a target point and speed for the active high-level state."""

        pos = obs["pos"].astype(np.float32)
        target_gate = int(obs["target_gate"])

        if self._state == KaFa1500State.TAKEOFF:
            target = np.array(
                [self._start_pos[0], self._start_pos[1], self._takeoff_height],
                dtype=np.float32,
            )
            if np.linalg.norm(target - pos) < self._planner_settings.takeoff_tol:
                if target_gate >= 0:
                    self._begin_scan()
                else:
                    self._state = KaFa1500State.FINISH
            return target, self._takeoff_heading(), self._action_settings.takeoff_speed

        if self._state == KaFa1500State.SCAN:
            if target_gate < 0:
                self._state = KaFa1500State.FINISH
            else:
                self._plan = self._navigator.plan_gate(obs, target_gate)
                if self._tick - self._scan_started_at >= self._planner_settings.scan_hold_steps:
                    self._state = KaFa1500State.APPROACH

            hover_target = pos.copy()
            hover_target[2] = max(hover_target[2], self._takeoff_height)
            return hover_target, self._takeoff_heading(), 0.0

        if self._state == KaFa1500State.FINISH or target_gate < 0:
            hold_target = pos.copy()
            hold_target[2] = max(hold_target[2], self._takeoff_height)
            return hold_target, self._action_builder.heading_vector(), 0.0

        if self._plan is None or target_gate != self._plan.gate_idx:
            self._begin_scan()
            return pos.copy(), self._takeoff_heading(), 0.0

        if self._state == KaFa1500State.APPROACH:
            path_target = self._navigator.follow_path(pos, self._plan)
            if self._navigator.segment_blocked(
                pos,
                path_target.target,
                obs["obstacles_pos"],
                self._planner_settings.obstacle_clearance,
            ):
                self._begin_scan()
                return pos.copy(), self._takeoff_heading(), 0.0

            if path_target.remaining <= self._planner_settings.approach_tol:
                self._state = KaFa1500State.PASS_GATE
                return (
                    self._plan.pass_target,
                    self._plan.gate_x,
                    self._action_settings.pass_speed,
                )

            final_corridor = path_target.remaining <= 0.30
            yaw_dir = self._plan.gate_x if final_corridor else path_target.yaw_dir
            speed = (
                self._action_settings.approach_speed
                if final_corridor
                else self._action_settings.route_speed
            )
            return path_target.target, yaw_dir.astype(np.float32), speed

        if self._state == KaFa1500State.PASS_GATE:
            signed_progress = float(np.dot(pos - self._plan.gate_pos, self._plan.gate_x))
            if signed_progress > self._planner_settings.replan_margin:
                if target_gate < 0:
                    self._state = KaFa1500State.FINISH
                else:
                    self._begin_scan()
            return self._plan.pass_target, self._plan.gate_x, self._action_settings.pass_speed

        raise ValueError(f"Invalid state: {self._state}")

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
