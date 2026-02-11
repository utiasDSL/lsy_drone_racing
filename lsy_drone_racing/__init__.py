"""LSY drone racing package for the Autonomous Drone Racing class @ TUM."""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Callable

from crazyflow.utils import enable_cache

import lsy_drone_racing.envs  # noqa: F401, register environments with gymnasium

logger = logging.getLogger(__name__)


def _compilation_cache_enabled(env_value: str | None) -> bool:
    """Return True unless env_value explicitly disables JAX compilation cache."""
    if env_value is None:
        return True
    return env_value.strip().lower() not in {"0", "false", "no", "off"}


def _resolve_cache_dir(cache_dir: str | None, machine: str | None = None) -> Path:
    """Resolve cache directory, defaulting to per-arch /tmp cache."""
    if cache_dir:
        return Path(cache_dir)
    machine_name = machine if machine is not None else platform.machine()
    return Path("/tmp") / f"jax_cache_{machine_name}"


def _maybe_enable_jax_compilation_cache(
    *,
    cache_toggle: str | None,
    cache_dir: str | None,
    cache_enabler: Callable[[Path], None],
    machine: str | None = None,
) -> tuple[bool, Path | None]:
    """Enable cache unless explicitly disabled via env."""
    if not _compilation_cache_enabled(cache_toggle):
        return False, None
    resolved_dir = _resolve_cache_dir(cache_dir, machine=machine)
    cache_enabler(resolved_dir)
    return True, resolved_dir


_cache_enabled, _cache_path = _maybe_enable_jax_compilation_cache(
    cache_toggle=os.environ.get("JAX_ENABLE_COMPILATION_CACHE"),
    cache_dir=os.environ.get("LSY_JAX_CACHE_DIR"),
    cache_enabler=lambda p: enable_cache(cache_path=p),
)
if _cache_enabled:
    logger.warning("JAX compilation cache enabled (dir=%s)", _cache_path)
else:
    logger.warning(
        "JAX compilation cache disabled via JAX_ENABLE_COMPILATION_CACHE=%r",
        os.environ.get("JAX_ENABLE_COMPILATION_CACHE"),
    )
