import random
import time
from sim import simulate

def run_multiple_simulations():
    # Run the simulation 10 times
    for run in range(1, 11):
        print(f"\n=== Starting simulation run {run}/10 ===")
        
        # Random wait to help ensure different seeds
        time.sleep(random.uniform(0.1, 0.5))
        
        # Run simulation with GUI disabled (gui=False) and random seed
        # You can set gui=True if you want to visualize each run
        ep_times = simulate(n_runs=1, gui=True)
        
        print(f"Run {run} completed with time: {ep_times[0] if ep_times[0] is not None else 'DNF'}")

if __name__ == "__main__":
    run_multiple_simulations()