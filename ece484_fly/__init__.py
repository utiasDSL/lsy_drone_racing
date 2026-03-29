from crazyflow.utils import enable_cache

import ece484_fly.envs  # noqa: F401, register environments with gymnasium

enable_cache()  # Enable persistent caching of jax functions
