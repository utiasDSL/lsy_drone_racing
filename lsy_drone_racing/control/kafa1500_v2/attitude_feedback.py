"""Closed-loop attitude feedback control."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.kafa1500_v2.utils import clip_norm, slew_angle

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_v2.config import FeedbackConfig
    from lsy_drone_racing.control.kafa1500_v2.types import Observation, Reference, Vec3


class AttitudeFeedback:
    """Convert position/velocity errors into `[roll, pitch, yaw, thrust]` commands."""

    def __init__(self, race_config: dict, config: FeedbackConfig, freq: float, yaw: float):
        """Load drone parameters and initialize filters."""
        self._config = config
        self._dt = 1.0 / float(freq)
        self._params = load_params(race_config.sim.physics, race_config.sim.drone_model)
        self._mass = float(self._params["mass"])
        self._thrust_min = float(self._params["thrust_min"] * 4.0)
        self._thrust_max = float(self._params["thrust_max"] * 4.0)
        self._integral = np.zeros(3, dtype=np.float32)
        self._last_action = np.array(
            [0.0, 0.0, yaw, self._mass * self._config.gravity], dtype=np.float32
        )

    def reset(self, yaw: float) -> None:
        """Reset integrator and output filters."""
        self._integral[:] = 0.0
        self._last_action[:] = np.array(
            [0.0, 0.0, yaw, self._mass * self._config.gravity], dtype=np.float32
        )

    def command(self, obs: Observation, reference: Reference) -> NDArray[np.float32]:
        """Compute the attitude-mode command."""
        pos = obs["pos"].astype(np.float32)
        vel = obs["vel"].astype(np.float32)
        pos_error = reference.position - pos
        vel_error = reference.velocity - vel

        self._integral = np.clip(
            self._integral + pos_error * self._dt,
            -self._config.integral_limit,
            self._config.integral_limit,
        ).astype(np.float32)

        ff_acc = clip_norm(reference.acceleration, self._config.max_feedforward_acc)
        force = (
            self._config.kp * pos_error
            + self._config.ki * self._integral
            + self._config.kd * vel_error
            + self._config.feedforward_acc_scale * self._mass * ff_acc
        ).astype(np.float32)
        force[2] += self._mass * self._config.gravity * self._config.hover_thrust_scale

        euler = self._force_to_euler(force, reference.yaw)
        thrust = self._force_to_thrust(force, obs["quat"])
        return self._smooth_and_clip(euler, thrust)

    def _force_to_thrust(self, force: Vec3, quat: NDArray[np.floating]) -> float:
        z_body = R.from_quat(quat).as_matrix()[:, 2].astype(np.float32)
        return float(np.clip(float(force @ z_body), self._thrust_min, self._thrust_max))

    def _force_to_euler(self, force: Vec3, yaw: float) -> NDArray[np.float32]:
        z_axis = force / (float(np.linalg.norm(force)) + 1e-6)
        x_heading = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        y_axis = np.cross(z_axis, x_heading)
        if float(np.linalg.norm(y_axis)) < 1e-6:
            y_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32)
        y_axis /= float(np.linalg.norm(y_axis)) + 1e-6
        x_axis = np.cross(y_axis, z_axis)
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        euler = R.from_matrix(rotation).as_euler("xyz", degrees=False).astype(np.float32)
        euler[:2] = np.clip(euler[:2], -self._config.max_tilt, self._config.max_tilt)
        return euler

    def _smooth_and_clip(self, euler: NDArray[np.float32], thrust: float) -> NDArray[np.float32]:
        action = self._last_action.copy()
        alpha = self._config.attitude_smoothing
        action[:2] = ((1.0 - alpha) * action[:2] + alpha * euler[:2]).astype(np.float32)
        action[2] = slew_angle(action[2], float(euler[2]), self._config.max_yaw_rate_step)
        beta = self._config.thrust_smoothing
        action[3] = float((1.0 - beta) * action[3] + beta * thrust)
        action[:2] = np.clip(action[:2], -self._config.max_tilt, self._config.max_tilt)
        action[2] = float(np.clip(action[2], -np.pi / 2.0, np.pi / 2.0))
        action[3] = float(np.clip(action[3], self._thrust_min, self._thrust_max))
        self._last_action = action.astype(np.float32)
        return self._last_action.copy()
