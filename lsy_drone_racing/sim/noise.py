"""Composable noise classes with selectable dimensions."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Noise:
    """Base class for noise applied to inputs or dyanmics."""

    def __init__(self, dim: int, mask: NDArray[np.bool] | None = None):
        """Initialize basic parameters.

        Args:
            dim: The dimensionality of the noise.
            mask: A boolean mask to apply the noise to only certain dimensions.
        """
        self.dim = dim
        self.np_random = np.random.default_rng()
        self.mask = np.asarray(mask) if mask is not None else np.ones(dim, dtype=bool)
        assert self.dim == len(self.mask), "Mask shape should be the same as dim."
        self._step = 0

    def reset(self):
        """Reset the noise to its initial state."""
        self._step = 0

    def step(self):
        """Increment the noise step for time dependent noise classes."""
        self._step += 1

    def apply(self, target: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply the noise to the target.

        Args:
            target: The target to apply the noise to. By default, no noise is applied.

        Returns:
            The noisy target.
        """
        return target

    def seed(self, seed: int | None = None):
        """Set the random number generator seed for the noise for deterministic behaviour.

        Args:
            seed: The seed to set the random number generator to. If None, the seed is random.
        """
        self.np_random = np.random.default_rng(seed)


class UniformNoise(Noise):
    """I.i.d uniform noise ~ U(low, high) per time step."""

    def __init__(
        self, dim: int, mask: NDArray[np.bool] | None = None, low: float = 0.0, high: float = 1.0
    ):
        """Initialize the uniform noise.

        Args:
            dim: The dimensionality of the noise.
            mask: A boolean mask to apply the noise to only certain dimensions.
            low: The lower bound of the uniform distribution.
            high: The upper bound of the uniform distribution.
        """
        super().__init__(dim, mask)
        assert isinstance(low, (float, list, np.ndarray)), "low must be float or list."
        assert isinstance(high, (float, list, np.ndarray)), "high must be float or list."
        self.low = np.array([low] * self.dim) if isinstance(low, float) else np.asarray(low)
        self.high = np.array([high] * self.dim) if isinstance(high, float) else np.asarray(high)

    def apply(self, target: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply the noise to the target.

        Args:
            target: The target to apply the noise to.

        Returns:
            The noisy target.
        """
        noise = self.np_random.uniform(self.low, self.high, size=self.dim)
        if self.mask is not None:
            noise *= self.mask
        return target + noise


class GaussianNoise(Noise):
    """I.i.d Gaussian noise per time step."""

    def __init__(
        self,
        dim: int,
        mask: NDArray[np.bool] | None = None,
        std: float | NDArray[np.floating] = 1.0,
    ):
        """Initialize the uniform noise.

        Args:
            dim: The dimensionality of the noise.
            mask: A boolean mask to apply the noise to only certain dimensions.
            std: The standard deviation of the distribution.
        """
        super().__init__(dim, mask)
        assert isinstance(std, (float, list, np.ndarray)), "std must be float or list."
        self.std = np.array([std] * self.dim) if isinstance(std, float) else np.asarray(std)
        assert self.dim == len(self.std), "std shape should be the same as dim."

    def apply(self, target: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply the noise to the target.

        Args:
            target: The target to apply the noise to.

        Returns:
            The noisy target.
        """
        noise = self.np_random.normal(0, self.std, size=self.dim)
        if self.mask is not None:
            noise *= self.mask
        return target + noise


class NoiseList(list):
    """Combine list of noises as one."""

    def __init__(self, noises: list[Noise]):
        """Initialization of the list of noises."""
        super().__init__(noises)

    def reset(self):
        """Sequentially reset noises."""
        for n in self:
            n.reset()

    def apply(self, target: NDArray[np.floating]) -> NDArray[np.floating]:
        """Sequentially apply noises to the target.

        Args:
            target: The target to apply the noise to.

        Returns:
            The noisy target.
        """
        noisy = target
        for n in self:
            noisy = n.apply(noisy)
        return noisy

    def seed(self, seed: int | None = None):
        """Reset seed from env.

        Args:
            seed: The seed to set the random number generator to. If None, the seed is random.
        """
        for n in self:
            n.seed(seed)

    @staticmethod
    def from_specs(noise_specs: list[dict]) -> NoiseList:
        """Create a NoiseList from a list of noise specifications.

        Args:
            noise_specs: List of dicts defining the noises info.
        """
        disturb_list = []
        # Each noise for the mode.
        for n_spec in noise_specs:
            assert isinstance(n_spec, dict), "Each noise must be specified as dict."
            assert "type" in n_spec.keys(), "Each noise must have a 'type' key."
            d_class = getattr(sys.modules[__name__], n_spec["type"])
            disturb_list.append(d_class(**{k: v for k, v in n_spec.items() if k != "type"}))
        return NoiseList(disturb_list)
