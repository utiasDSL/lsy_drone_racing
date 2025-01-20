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


def main(
    n_tests: int = 2,
    number: int = 100,
    multi_drone: bool = False,
    reset: bool = True,
    step: bool = True,
):
    reset_fn, step_fn = time_sim_reset, time_sim_step
    if multi_drone:
        reset_fn, step_fn = time_multi_drone_reset, time_multi_drone_step
    if reset:
        timings = reset_fn(n_tests=n_tests, number=number)
        print_benchmark_results(name="Racing env reset", timings=timings / number)
    if step:
        timings = step_fn(n_tests=n_tests, number=number)
        print_benchmark_results(name="Racing env steps", timings=timings / number)
        timings = step_fn(n_tests=n_tests, number=number, physics_mode="sys_id")
        print_benchmark_results(name="Racing env steps (sys_id backend)", timings=timings / number)
        # timings = step_fn(n_tests=n_tests, number=number, physics_mode="mujoco")
        # print_benchmark_results(name="Sim steps (mujoco backend)", timings=timings / number)


if __name__ == "__main__":
    fire.Fire(main)
