from __future__ import annotations

import numpy as np
from sim import time_sim_attitude_step, time_sim_reset, time_sim_step


def print_benchmark_results(name: str, timings: list[float]):
    print(f"\nResults for {name}:")
    print(f"Mean/std: {np.mean(timings):.2e}s +- {np.std(timings):.2e}s")
    print(f"Min time: {np.min(timings):.2e}s")
    print(f"Max time: {np.max(timings):.2e}s")
    print(f"FPS: {1 / np.mean(timings):.2f}")


if __name__ == "__main__":
    n_tests = 10
    sim_steps = 10
    timings = time_sim_reset(n_tests=n_tests)
    print_benchmark_results(name="Sim reset", timings=timings)
    timings = time_sim_step(n_tests=n_tests, sim_steps=sim_steps)
    print_benchmark_results(name="Sim steps", timings=timings / sim_steps)
    timings = time_sim_step(n_tests=n_tests, sim_steps=sim_steps, physics_mode="dyn")
    print_benchmark_results(name="Sim steps (dyn backend)", timings=timings / sim_steps)
    timings = time_sim_step(n_tests=n_tests, sim_steps=sim_steps, physics_mode="pyb_gnd")
    print_benchmark_results(name="Sim steps (pyb_gnd backend)", timings=timings / sim_steps)
    timings = time_sim_step(n_tests=n_tests, sim_steps=sim_steps, physics_mode="pyb_drag")
    print_benchmark_results(name="Sim steps (pyb_drag backend)", timings=timings / sim_steps)
    timings = time_sim_step(n_tests=n_tests, sim_steps=sim_steps, physics_mode="pyb_dw")
    print_benchmark_results(name="Sim steps (pyb_dw backend)", timings=timings / sim_steps)
    timings = time_sim_step(n_tests=n_tests, sim_steps=sim_steps, physics_mode="pyb_gnd_drag_dw")
    print_benchmark_results(name="Sim steps (pyb_gnd_drag_dw backend)", timings=timings / sim_steps)
    timings = time_sim_attitude_step(n_tests=n_tests, sim_steps=sim_steps)
    print_benchmark_results(name="Sim steps (sys_id backend)", timings=timings / sim_steps)
