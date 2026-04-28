"""Closed-loop reference target management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.kafa1500_v2.types import Reference

if TYPE_CHECKING:
    from lsy_drone_racing.control.kafa1500_v2.config import ReferenceConfig
    from lsy_drone_racing.control.kafa1500_v2.types import CubicPath, Vec3


class ReferenceManager:
    """Advance the active path index based on observed position, not time."""

    def __init__(self, config: ReferenceConfig):
        """Initialize the manager."""
        self._config = config
        self._path: CubicPath | None = None
        self._index = config.start_index
        self._last_advance_tick = -config.min_ticks_between_advances
        self._last_yaw = 0.0

    @property
    def path(self) -> CubicPath | None:
        """Return the active path."""
        return self._path

    @property
    def index(self) -> int:
        """Return the active sample index."""
        return self._index

    def reset(self, path: CubicPath, yaw: float) -> None:
        """Load a new cubic path and reset target advancement."""
        self._path = path
        self._index = min(max(0, self._config.start_index), len(path.points) - 1)
        self._last_advance_tick = -self._config.min_ticks_between_advances
        self._last_yaw = yaw

    def update(self, pos: Vec3, tick: int) -> Reference:
        """Return the active reference, advancing only when close enough."""
        if self._path is None:
            return self.hold(pos)

        self._advance_to_nearby_forward_sample(pos, tick)
        for _ in range(self._config.max_advance_per_step):
            distance = float(np.linalg.norm(pos - self._path.points[self._index]))
            if not self._can_advance(distance, tick):
                break
            if self._index >= len(self._path.points) - 1:
                break
            self._index += 1
            self._last_advance_tick = tick

        distance = float(np.linalg.norm(pos - self._path.points[self._index]))
        return self._reference(self._index, distance)

    def hold(self, pos: Vec3) -> Reference:
        """Hold the current position."""
        zero = np.zeros(3, dtype=np.float32)
        return Reference(
            position=pos.astype(np.float32),
            velocity=zero,
            acceleration=zero,
            yaw=self._last_yaw,
            index=self._index,
            distance=0.0,
            done=True,
        )

    def _can_advance(self, distance: float, tick: int) -> bool:
        if tick - self._last_advance_tick < self._config.min_ticks_between_advances:
            return False
        threshold = self._config.target_reached_distance
        if self._last_advance_tick == tick:
            threshold -= self._config.target_hysteresis
        return distance <= threshold

    def _advance_to_nearby_forward_sample(self, pos: Vec3, tick: int) -> None:
        if self._path is None:
            return
        if tick - self._last_advance_tick < self._config.min_ticks_between_advances:
            return
        stop = min(len(self._path.points), self._index + self._config.nearest_forward_search + 1)
        candidates = self._path.points[self._index:stop]
        distances = np.linalg.norm(candidates - pos, axis=1)
        offset = int(np.argmin(distances))
        if offset <= 0:
            return
        if float(distances[offset]) <= self._config.target_reached_distance:
            self._index += min(offset, self._config.max_advance_per_step)
            self._last_advance_tick = tick

    def _reference(self, index: int, distance: float) -> Reference:
        if self._path is None:
            raise RuntimeError("Reference requested without an active path.")

        param = float(self._path.params[index])
        position = np.asarray(self._path.spline(param), dtype=np.float32)
        tangent = np.asarray(self._path.velocity_spline(param), dtype=np.float32)
        curvature = np.asarray(self._path.acceleration_spline(param), dtype=np.float32)
        tangent_norm = float(np.linalg.norm(tangent))
        speed = self._speed_for_index(index)

        if tangent_norm < 1e-6:
            velocity = np.zeros(3, dtype=np.float32)
            acceleration = np.zeros(3, dtype=np.float32)
        else:
            ds_dt = speed / tangent_norm
            velocity = (tangent * ds_dt).astype(np.float32)
            acceleration = (curvature * ds_dt**2).astype(np.float32)
            if self._config.follow_path_yaw and float(np.linalg.norm(tangent[:2])) > 1e-6:
                self._last_yaw = float(np.arctan2(tangent[1], tangent[0]))

        return Reference(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            yaw=self._last_yaw,
            index=index,
            distance=distance,
            done=index >= len(self._path.points) - 1,
        )

    def _speed_for_index(self, index: int) -> float:
        if self._path is None:
            return 0.0
        if index >= len(self._path.points) - 1:
            return self._config.final_speed
        gate_id = int(self._path.gate_indices[index])
        if gate_id >= 0:
            gate_window = self._path.gate_indices[
                max(0, index - self._config.gate_window_samples) : index
                + self._config.gate_window_samples
                + 1
            ]
            if np.any(gate_window == gate_id):
                return self._config.gate_speed
        return self._config.nominal_speed
