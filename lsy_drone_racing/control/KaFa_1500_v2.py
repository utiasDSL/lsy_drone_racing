"""KaFa 1500 v2: Kaan spline generation with attitude path following."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.kafa1500_v2.config import (
    FeedbackConfig,
    PathConfig,
    ReferenceConfig,
)
from lsy_drone_racing.control.kafa1500_v2.gate_targets import GateTargetPlanner
from lsy_drone_racing.control.kafa1500_v2.path_follower import PathFollower
from lsy_drone_racing.control.kafa1500_v2.reference_adapter import ReferenceAdapter
from lsy_drone_racing.control.kafa1500_v2.spline_path import CubicPathBuilder
from lsy_drone_racing.control.kafa1500_v2.types import FlightPhase, Reference

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_v2.types import Observation


class KaFa1500V2(Controller):
    """Closed-loop attitude controller using Kaan spline geometry."""

    def __init__(self, obs: Observation, info: dict, config: dict):
        """Initialize controller modules."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("KaFa_1500_v2 requires env.control_mode = 'attitude'.")

        self._path_config = PathConfig()
        self._reference_config = ReferenceConfig()
        self._feedback_config = FeedbackConfig()
        self._target_planner = GateTargetPlanner(self._path_config)
        self._path_builder = CubicPathBuilder(self._path_config)
        self._adapter = ReferenceAdapter(self._reference_config.gate_window_samples)

        self._freq = float(config.env.freq)
        self._tick = 0
        self._finished = False
        self._phase = FlightPhase.TAKEOFF
        self._last_target_gate = int(obs["target_gate"])
        self._last_gates_visited = np.asarray(obs["gates_visited"]).copy()
        self._last_obstacles_visited = np.asarray(obs["obstacles_visited"]).copy()
        self._deferred_replan_gate: int | None = None
        self._deferred_replan_until = 0
        self._yaw0 = self._yaw_from_obs(obs)
        self._takeoff_position = self._make_takeoff_position(obs)
        self._active_reference = self._takeoff_reference(obs)
        self._follower = PathFollower(
            config,
            self._reference_config,
            self._feedback_config,
            self._freq,
            self._yaw0,
        )

    def compute_control(self, obs: Observation, info: dict | None = None) -> NDArray[np.floating]:
        """Compute the current attitude command from observed tracking error."""
        target_gate = int(obs["target_gate"])
        if target_gate < 0:
            self._phase = FlightPhase.FINISH
            self._finished = True
            self._active_reference = self._follower.hold(obs)
            return self._follower.command_reference(obs, self._active_reference)

        if self._phase == FlightPhase.TAKEOFF:
            self._active_reference = self._takeoff_reference(obs)
            if self._active_reference.distance <= self._path_config.takeoff_reached_distance:
                self._replan(obs)
                self._phase = FlightPhase.TRACK
                action = self._follower.command_path(obs, self._tick)
                active = self._follower.active_reference
                if active is not None:
                    self._active_reference = active
                return action
            return self._follower.command_reference(obs, self._active_reference)

        if self._should_replan(obs):
            self._replan(obs)

        self._remember_observation_flags(obs)
        action = self._follower.command_path(obs, self._tick)
        active = self._follower.active_reference
        if active is not None:
            self._active_reference = active
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
        """Advance internal bookkeeping."""
        self._tick += 1
        self._finished = bool(self._finished or terminated or truncated or obs["target_gate"] == -1)
        return self._finished

    def episode_callback(self) -> None:
        """Reset per-episode state."""
        self._tick = 0
        self._finished = False
        self._phase = FlightPhase.TAKEOFF
        self._follower.reset_feedback(self._yaw0)
        self._deferred_replan_gate = None
        self._deferred_replan_until = 0

    def render_callback(self, sim: Sim) -> None:
        """Visualize adapted cubic path and active target."""
        path = self._follower.path
        if path is not None:
            draw_line(sim, path.points, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(
            sim,
            self._active_reference.position.reshape(1, -1),
            rgba=(1.0, 0.0, 0.0, 1.0),
            size=0.03,
        )

    def _replan(self, obs: Observation) -> None:
        """Build a Kaan-style cubic path and adapt it for the attitude follower."""
        controls, gate_ids = self._target_planner.build_control_points(
            obs,
            obs["pos"].astype(np.float32),
        )
        raw_path = self._path_builder.build(controls, gate_ids)
        self._target_planner.validate_gate_clearance(raw_path.points, obs)
        path = self._adapter.adapt(raw_path)
        self._follower.reset_path(path, yaw=self._yaw_from_obs(obs))

    def _should_replan(self, obs: Observation) -> bool:
        """Replan when newly observed track state changes the reliable plan."""
        target_gate = int(obs["target_gate"])
        target_changed = target_gate != self._last_target_gate
        gates_visited = np.asarray(obs["gates_visited"])
        obstacles_visited = np.asarray(obs["obstacles_visited"])
        new_gate_seen = bool((gates_visited & ~self._last_gates_visited).any())
        new_obstacle_seen = bool((obstacles_visited & ~self._last_obstacles_visited).any())
        if self._deferred_replan_gate == target_gate:
            if self._tick < self._deferred_replan_until:
                return False
            self._deferred_replan_gate = None
            return True
        if not target_changed:
            return new_gate_seen or new_obstacle_seen
        if self._active_path_contains_gate(target_gate):
            self._deferred_replan_gate = target_gate
            self._deferred_replan_until = self._tick + 20
            return False
        return True

    def _active_path_contains_gate(self, target_gate: int) -> bool:
        """Return whether the adapted path already includes the new target gate."""
        path = self._follower.path
        if path is None or target_gate < 0:
            return False
        return bool(np.any(path.gate_indices[self._follower.index :] == target_gate))

    def _remember_observation_flags(self, obs: Observation) -> None:
        """Store observations after replanning decisions have used the previous values."""
        self._last_target_gate = int(obs["target_gate"])
        self._last_gates_visited = np.asarray(obs["gates_visited"]).copy()
        self._last_obstacles_visited = np.asarray(obs["obstacles_visited"]).copy()

    def _make_takeoff_position(self, obs: Observation) -> NDArray[np.float32]:
        """Create a vertical takeoff target."""
        pos = obs["pos"].astype(np.float32)
        target = pos.copy()
        target[2] = max(self._path_config.takeoff_height, float(pos[2] + 0.55))
        return target

    def _takeoff_reference(self, obs: Observation) -> Reference:
        """Hold XY while climbing to a safe height."""
        pos = obs["pos"].astype(np.float32)
        zero = np.zeros(3, dtype=np.float32)
        return Reference(
            position=self._takeoff_position.copy(),
            velocity=zero,
            acceleration=zero,
            yaw=self._yaw0,
            index=0,
            distance=float(np.linalg.norm(pos - self._takeoff_position)),
            done=False,
        )

    @staticmethod
    def _yaw_from_obs(obs: Observation) -> float:
        """Extract yaw from the drone quaternion."""
        return float(R.from_quat(obs["quat"]).as_euler("xyz", degrees=False)[2])
