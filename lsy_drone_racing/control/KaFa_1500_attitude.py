"""Gate-aware closed-loop attitude controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.kafa1500_attitude.attitude_feedback import AttitudeFeedback
from lsy_drone_racing.control.kafa1500_attitude.config import (
    FeedbackConfig,
    PathConfig,
    ReferenceConfig,
)
from lsy_drone_racing.control.kafa1500_attitude.gate_targets import GateTargetPlanner
from lsy_drone_racing.control.kafa1500_attitude.reference_manager import ReferenceManager
from lsy_drone_racing.control.kafa1500_attitude.spline_path import CubicPathBuilder
from lsy_drone_racing.control.kafa1500_attitude.types import FlightPhase, Reference

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_attitude.types import Observation


class KaFa1500Attitude(Controller):
    """Closed-loop attitude controller using gate-aware cubic spline references."""

    def __init__(self, obs: Observation, info: dict, config: dict):
        """Initialize controller modules."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("KaFa_1500_attitude requires env.control_mode = 'attitude'.")

        self._path_config = PathConfig()
        self._reference_config = ReferenceConfig()
        self._feedback_config = FeedbackConfig()
        self._target_planner = GateTargetPlanner(self._path_config)
        self._path_builder = CubicPathBuilder(self._path_config)
        self._references = ReferenceManager(self._reference_config)

        self._freq = float(config.env.freq)
        self._tick = 0
        self._finished = False
        self._phase = FlightPhase.TAKEOFF
        self._last_target_gate = int(obs["target_gate"])
        self._last_gates_visited = np.asarray(obs["gates_visited"]).copy()
        self._last_obstacles_visited = np.asarray(obs["obstacles_visited"]).copy()
        self._yaw0 = self._yaw_from_obs(obs)
        self._takeoff_position = self._make_takeoff_position(obs)
        self._active_reference = self._takeoff_reference(obs)
        self._feedback = AttitudeFeedback(config, self._feedback_config, self._freq, self._yaw0)

    def compute_control(
        self, obs: Observation, info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the current attitude command from observed tracking error."""
        target_gate = int(obs["target_gate"])
        if target_gate < 0:
            self._phase = FlightPhase.FINISH
            self._finished = True
            self._active_reference = self._references.hold(obs["pos"].astype(np.float32))
            return self._feedback.command(obs, self._active_reference)

        if self._phase == FlightPhase.TAKEOFF:
            self._active_reference = self._takeoff_reference(obs)
            if self._active_reference.distance <= self._path_config.takeoff_reached_distance:
                self._replan(obs)
                self._phase = FlightPhase.TRACK
            return self._feedback.command(obs, self._active_reference)

        if self._should_replan(obs):
            self._replan(obs)

        self._remember_observation_flags(obs)
        self._active_reference = self._references.update(obs["pos"].astype(np.float32), self._tick)
        return self._feedback.command(obs, self._active_reference)

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
        self._feedback.reset(self._yaw0)

    def render_callback(self, sim: Sim) -> None:
        """Visualize cubic path and active target."""
        path = self._references.path
        if path is not None:
            draw_line(sim, path.points, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(
            sim,
            self._active_reference.position.reshape(1, -1),
            rgba=(1.0, 0.0, 0.0, 1.0),
            size=0.03,
        )

    def _replan(self, obs: Observation) -> None:
        """Build a new gate-aware cubic path from the current drone state."""
        controls, gate_ids = self._target_planner.build_control_points(
            obs, obs["pos"].astype(np.float32)
        )
        path = self._path_builder.build(controls, gate_ids)
        self._target_planner.validate_gate_clearance(path.points, obs)
        self._references.reset(path, yaw=self._yaw_from_obs(obs))
        self._active_reference = self._references.update(obs["pos"].astype(np.float32), self._tick)

    def _should_replan(self, obs: Observation) -> bool:
        """Replan when the environment reveals new gate/obstacle state or gate index changes."""
        target_changed = int(obs["target_gate"]) != self._last_target_gate
        gates_visited = np.asarray(obs["gates_visited"])
        obstacles_visited = np.asarray(obs["obstacles_visited"])
        new_gate_seen = bool((gates_visited & ~self._last_gates_visited).any())
        new_obstacle_seen = bool((obstacles_visited & ~self._last_obstacles_visited).any())
        return target_changed or new_gate_seen or new_obstacle_seen

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
