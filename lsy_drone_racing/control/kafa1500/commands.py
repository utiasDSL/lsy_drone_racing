"""Full-state action generation for the KaFa1500 controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.kafa1500.types import KaFa1500State, Observation, Vec3

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500.settings import ActionSettings


class StateActionBuilder:
    """Convert a position target into a filtered 13D full-state action."""

    def __init__(self, settings: ActionSettings, takeoff_yaw: float):
        """Store tuning parameters and initialize filter state."""
        self._settings = settings
        self._takeoff_yaw = takeoff_yaw
        self._last_yaw = takeoff_yaw
        self._filtered_vel_cmd = np.zeros(3, dtype=np.float32)
        self._filtered_acc_cmd = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        """Reset filters and restore the takeoff heading."""
        self._last_yaw = self._takeoff_yaw
        self._filtered_vel_cmd[:] = 0.0
        self._filtered_acc_cmd[:] = 0.0

    def heading_vector(self) -> Vec3:
        """Return the current heading direction in the XY plane."""
        return np.array([np.cos(self._last_yaw), np.sin(self._last_yaw), 0.0], dtype=np.float32)

    def build(
        self,
        obs: Observation,
        target_pos: Vec3,
        yaw_dir: Vec3,
        speed_limit: float,
        state: KaFa1500State,
        state_changed: bool,
        ref_vel: Vec3 | None = None,
        ref_acc: Vec3 | None = None,
    ) -> NDArray[np.float32]:
        """Generate the next 13D full-state command."""
        pos = obs["pos"].astype(np.float32)
        vel = obs["vel"].astype(np.float32)
        pos_error = target_pos - pos

        desired_vel = self._desired_velocity(
            pos_error,
            vel,
            yaw_dir,
            speed_limit,
            state,
            ref_vel,
        )
        desired_vel = self._filter_velocity(desired_vel, speed_limit, state, state_changed)

        desired_acc = self._desired_acceleration(pos_error, vel, desired_vel, state, ref_acc)
        desired_acc = self._filter_acceleration(desired_acc, state, state_changed)

        self._update_yaw(desired_vel, yaw_dir, state)

        action = np.zeros(13, dtype=np.float32)
        action[0:3] = target_pos
        #action[3:6] = desired_vel/20
        #action[6:9] = desired_acc/20
        action[9] = self._last_yaw
        action[1] -= 0.2  # Small vertical bias to encourage better ground clearance
        return action

    @staticmethod
    def yaw_from_quat(quat: NDArray[np.floating]) -> float:
        """Extract yaw from an xyzw quaternion."""
        return float(R.from_quat(quat).as_euler("xyz", degrees=False)[2])

    def _desired_velocity(
        self,
        pos_error: Vec3,
        vel: Vec3,
        yaw_dir: Vec3,
        speed_limit: float,
        state: KaFa1500State,
        ref_vel: Vec3 | None,
    ) -> Vec3:
        """Compute the raw desired velocity before filtering."""
        desired_vel = self._settings.pos_gain * pos_error - self._settings.vel_damping * vel
        if state in (KaFa1500State.APPROACH, KaFa1500State.PASS_GATE):
            correction_vel = (
                self._settings.tangent_correction_gain * pos_error
                - self._settings.tangent_velocity_damping * vel
            )
            if ref_vel is not None and np.linalg.norm(ref_vel) > 1e-6:
                desired_vel = ref_vel.astype(np.float32) + correction_vel
            else:
                tangent = yaw_dir.astype(np.float32)
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm > 1e-6:
                    tangent /= tangent_norm
                    desired_vel = tangent * speed_limit + correction_vel

        desired_vel = self._clip_norm(desired_vel.astype(np.float32), speed_limit)
        if state == KaFa1500State.TAKEOFF:
            desired_vel[:2] = 0.0
        elif state == KaFa1500State.SCAN:
            desired_vel[:2] *= 0.2
        return desired_vel.astype(np.float32)

    def _filter_velocity(
        self,
        desired_vel: Vec3,
        speed_limit: float,
        state: KaFa1500State,
        state_changed: bool,
    ) -> Vec3:
        """Low-pass filter the velocity command outside hover-like states."""
        if state_changed or state in (KaFa1500State.TAKEOFF, KaFa1500State.SCAN):
            self._filtered_vel_cmd = desired_vel.copy()
        else:
            alpha = self._settings.velocity_filter_gain
            self._filtered_vel_cmd = (
                (1.0 - alpha) * self._filtered_vel_cmd + alpha * desired_vel
            ).astype(np.float32)
        return self._clip_norm(self._filtered_vel_cmd, speed_limit)

    def _desired_acceleration(
        self,
        pos_error: Vec3,
        vel: Vec3,
        desired_vel: Vec3,
        state: KaFa1500State,
        ref_acc: Vec3 | None,
    ) -> Vec3:
        """Compute the raw desired acceleration before filtering."""
        feedforward_acc = np.zeros(3, dtype=np.float32)
        if ref_acc is not None and np.linalg.norm(ref_acc) > 1e-6:
            feedforward_acc = ref_acc.astype(np.float32)

        desired_acc = feedforward_acc + self._settings.acc_gain * (desired_vel - vel)
        desired_acc[2] += self._settings.vertical_acc_gain * pos_error[2]
        desired_acc[:2] = self._clip_norm(desired_acc[:2], self._lateral_acc_cap(state))
        desired_acc[2] = float(
            np.clip(
                desired_acc[2],
                -self._settings.max_vertical_acc,
                self._settings.max_vertical_acc,
            )
        )

        if state == KaFa1500State.TAKEOFF:
            desired_acc[:2] = self._clip_norm(
                -1.5 * vel[:2],
                self._settings.max_takeoff_lateral_acc,
            )
        elif state == KaFa1500State.SCAN:
            desired_acc[:2] = self._clip_norm(-1.0 * vel[:2], self._settings.max_scan_lateral_acc)

        return desired_acc.astype(np.float32)

    def _filter_acceleration(
        self,
        desired_acc: Vec3,
        state: KaFa1500State,
        state_changed: bool,
    ) -> Vec3:
        """Low-pass filter the acceleration command outside hover-like states."""
        if state_changed or state in (KaFa1500State.TAKEOFF, KaFa1500State.SCAN):
            self._filtered_acc_cmd = desired_acc.copy()
        else:
            beta = self._settings.accel_filter_gain
            self._filtered_acc_cmd = (
                (1.0 - beta) * self._filtered_acc_cmd + beta * desired_acc
            ).astype(np.float32)
        return self._filtered_acc_cmd

    def _update_yaw(self, desired_vel: Vec3, yaw_dir: Vec3, state: KaFa1500State) -> None:
        """Update the commanded yaw from the intended planar travel direction."""
        planar_dir = yaw_dir[:2] if np.linalg.norm(yaw_dir[:2]) > 1e-6 else desired_vel[:2]
        if state in (KaFa1500State.TAKEOFF, KaFa1500State.SCAN):
            self._last_yaw = self._takeoff_yaw
        elif np.linalg.norm(planar_dir) > 1e-6:
            self._last_yaw = float(np.arctan2(planar_dir[1], planar_dir[0]))

    def _lateral_acc_cap(self, state: KaFa1500State) -> float:
        """Select the lateral acceleration cap for the current state."""
        match state:
            case KaFa1500State.TAKEOFF:
                return self._settings.max_takeoff_lateral_acc
            case KaFa1500State.SCAN:
                return self._settings.max_scan_lateral_acc
            case KaFa1500State.APPROACH:
                return self._settings.max_approach_lateral_acc
            case KaFa1500State.PASS_GATE:
                return self._settings.max_pass_lateral_acc
            case _:
                return self._settings.max_route_lateral_acc

    @staticmethod
    def _clip_norm(vec: Vec3, max_norm: float) -> Vec3:
        """Scale a vector down to the requested norm limit."""
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6 or norm <= max_norm:
            return vec.astype(np.float32)
        return (vec * (max_norm / norm)).astype(np.float32)
