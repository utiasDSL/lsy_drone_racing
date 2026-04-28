"""Closed-loop path following composed from attitude reference and feedback logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.kafa1500_v2.attitude_feedback import AttitudeFeedback
from lsy_drone_racing.control.kafa1500_v2.reference_manager import ReferenceManager

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.kafa1500_v2.config import FeedbackConfig, ReferenceConfig
    from lsy_drone_racing.control.kafa1500_v2.types import CubicPath, Observation, Reference, Vec3


class PathFollower:
    """Feed adapted cubic references into the working attitude feedback controller."""

    def __init__(
        self,
        race_config: dict,
        reference_config: ReferenceConfig,
        feedback_config: FeedbackConfig,
        freq: float,
        yaw: float,
    ):
        """Initialize reference manager and feedback controller."""
        self._references = ReferenceManager(reference_config)
        self._feedback = AttitudeFeedback(race_config, feedback_config, freq, yaw)
        self._active_reference: Reference | None = None

    @property
    def path(self) -> CubicPath | None:
        """Return the current adapted path."""
        return self._references.path

    @property
    def index(self) -> int:
        """Return the current progress index."""
        return self._references.index

    @property
    def active_reference(self) -> Reference | None:
        """Return the last reference sent to the attitude feedback controller."""
        return self._active_reference

    def reset_feedback(self, yaw: float) -> None:
        """Reset integrators and output smoothing in the attitude controller."""
        self._feedback.reset(yaw)

    def reset_path(self, path: CubicPath, yaw: float) -> None:
        """Load a new adapted path into the progress-based reference manager."""
        self._references.reset(path, yaw)

    def command_reference(self, obs: Observation, reference: Reference) -> NDArray[np.float32]:
        """Command an explicit reference, used for takeoff and hold phases."""
        self._active_reference = reference
        return self._feedback.command(obs, reference)

    def command_path(self, obs: Observation, tick: int) -> NDArray[np.float32]:
        """Advance by observed progress and command the active path reference."""
        self._active_reference = self._references.update(obs["pos"].astype(np.float32), tick)
        return self._feedback.command(obs, self._active_reference)

    def hold(self, obs: Observation) -> Reference:
        """Hold the current observed position."""
        self._active_reference = self._references.hold(obs["pos"].astype(np.float32))
        return self._active_reference
