from __future__ import annotations

import fire
import numpy as np
from sim import time_multi_drone_reset, time_multi_drone_step, time_sim_reset, time_sim_step


def print_benchmark_results(name: str, timings: list[float]):
    print(f"\nResults for {name}:")
    print(f"Mean/std: {np.mean(timings):.2e}s +- {np.std(timings):.2e}s")
    print(f"Min time: {np.min(timings):.2e}s")
    print(f"Max time: {np.max(timings):.2e}s")
    print(f"FPS: {1 / np.mean(timings):.2f}")


def main(n_tests: int = 10, sim_steps: int = 10, multi_drone: bool = False):
    reset_fn, step_fn = time_sim_reset, time_sim_step
    if multi_drone:
        reset_fn, step_fn = time_multi_drone_reset, time_multi_drone_step
    timings = reset_fn(n_tests=n_tests)
    print_benchmark_results(name="Sim reset", timings=timings)
    timings = step_fn(n_tests=n_tests, sim_steps=sim_steps)
    print_benchmark_results(name="Sim steps", timings=timings / sim_steps)
    timings = step_fn(n_tests=n_tests, sim_steps=sim_steps, physics_mode="sys_id")
    print_benchmark_results(name="Sim steps (sys_id backend)", timings=timings / sim_steps)
    timings = step_fn(n_tests=n_tests, sim_steps=sim_steps, physics_mode="mujoco")
    print_benchmark_results(name="Sim steps (mujoco backend)", timings=timings / sim_steps)
    timings = step_fn(n_tests=n_tests, sim_steps=sim_steps, physics_mode="sys_id")
    print_benchmark_results(name="Sim steps (sys_id backend)", timings=timings / sim_steps)


if __name__ == "__main__":
    fire.Fire(main)
