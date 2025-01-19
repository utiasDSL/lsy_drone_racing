import jax
import numpy as np
import time
from lsy_drone_racing.envs.vec_drone_race2 import VectorMultiDroneRaceEnv


def analyze_timings(times: list[float], n_steps: int, n_worlds: int, freq: float) -> None:
    """Analyze timing results and print performance metrics."""
    if not times:
        raise ValueError("The list of timing results is empty.")

    tmin, idx_tmin = np.min(times), np.argmin(times)
    tmax, idx_tmax = np.max(times), np.argmax(times)

    # Check for significant variance
    if tmax / tmin > 10:
        print("Warning: Fn time varies by more than 10x. Is JIT compiling during the benchmark?")
        print(f"Times: max {tmax:.2e} @ {idx_tmax}, min {tmin:.2e} @ {idx_tmin}")

    # Performance metrics
    n_frames = n_steps * n_worlds  # Number of frames simulated
    total_time = np.sum(times)
    avg_step_time = np.mean(times)
    step_time_std = np.std(times)
    fps = n_frames / total_time
    real_time_factor = (n_steps / freq) * n_worlds / total_time

    print(
        f"Avg fn time: {avg_step_time:.2e}s, std: {step_time_std:.2e}"
        f"\nFPS: {fps:.3e}, Real time factor: {real_time_factor:.2e}\n"
    )


def profile_reset(sim, n_steps: int, device: str):
    """Profile the Crazyflow simulator reset performance."""
    times = []
    times_masked = []
    device = jax.devices(device)[0]

    # Ensure JIT compiled reset
    sim.reset()
    jax.block_until_ready(sim.data)

    # Test full reset
    for _ in range(n_steps):
        tstart = time.perf_counter()
        sim.reset()
        jax.block_until_ready(sim.data)
        times.append(time.perf_counter() - tstart)

    # Test masked reset (only reset first world)
    mask = np.zeros(sim.n_worlds, dtype=bool)
    mask[0] = True
    sim.reset(mask)
    jax.block_until_ready(sim.data)

    for _ in range(n_steps):
        tstart = time.perf_counter()
        sim.reset(mask)
        times_masked.append(time.perf_counter() - tstart)

    print("Sim reset performance:")
    analyze_timings(times, n_steps, sim.n_worlds, sim.freq)
    print("Sim masked reset performance:")
    analyze_timings(times_masked, n_steps, sim.n_worlds, sim.freq)


def profile_step(sim, n_steps: int, device: str):
    """Profile the Crazyflow simulator step performance."""
    times = []
    device = jax.devices(device)[0]

    sim.step(sim.freq // env.freq)
    jax.block_until_ready(sim.data)

    for _ in range(n_steps):
        tstart = time.perf_counter()
        sim.step(sim.freq // env.freq)
        times.append(time.perf_counter() - tstart)

    print("Sim step performance:")
    analyze_timings(times, n_steps, sim.n_worlds, env.freq)


def profile_env_step(env, n_steps: int, device: str):
    """Profile the environment step performance."""
    times = []
    device = jax.devices(device)[0]
    action = env.action_space.sample()
    env.step(action)

    for _ in range(n_steps):
        tstart = time.perf_counter()
        env.step(action)
        times.append(time.perf_counter() - tstart)

    print("Env step performance:")
    analyze_timings(times, n_steps, env.sim.n_worlds, env.freq)


from lsy_drone_racing.utils import load_config
from pathlib import Path

config = load_config(Path(__file__).parent / "config/multi_level3.toml")

env = VectorMultiDroneRaceEnv(
    1,
    config.env.n_drones,
    config.env.freq,
    config.sim,
    sensor_range=config.env.sensor_range,
    track=config.env.track,
    disturbances=config.env.disturbances,
    randomizations=config.env.randomizations,
    random_resets=config.env.random_resets,
    seed=config.env.seed,
)
# profile_reset(env.sim, 100, "cpu")
# profile_step(env.sim, 100, "cpu")
profile_env_step(env, 100, "cpu")
