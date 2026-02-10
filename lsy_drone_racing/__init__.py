"""LSY drone racing package for the Autonomous Drone Racing class @ TUM."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from crazyflow.utils import enable_cache

import lsy_drone_racing.envs  # noqa: F401, register environments with gymnasium

# Enable persistent caching of JAX functions.
#
# We pick a per-architecture cache dir by default to avoid loading cache artifacts
# built under a different CPU target (e.g., Rosetta vs native), which can trigger
# noisy XLA warnings and in the worst case illegal instructions.
_cache_dir = os.environ.get("LSY_JAX_CACHE_DIR")
if not _cache_dir:
    _cache_dir = str(Path("/tmp") / f"jax_cache_{platform.machine()}")
enable_cache(cache_path=Path(_cache_dir))
