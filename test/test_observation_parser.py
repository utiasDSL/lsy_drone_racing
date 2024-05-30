"""Script to test the observation parser module."""
import numpy as np

from lsy_drone_racing.env_modifiers import ObservationParser


def test_observation_parser():
    """Test the observation parser."""
    n_gates = 3
    n_obstacles = 2
    obs_parser = ObservationParser(n_gates, n_obstacles)

    # Check the uninitialized state
    assert obs_parser.uninitialized()
    if obs_parser.uninitialized():
        print("Uninitialized function is correct.")
    else:
        raise ValueError("Uninitialized function is incorrect.")

    # Check the observable variables
    obs_parser.drone_pos = np.array([1, 2, 3])
    obs_parser.drone_yaw = np.pi
    obs_parser.gates_pos = np.array([[1, 2, 3], [1, 1, 1], [1, 1, 1]])
    obs_parser.gates_yaw = np.array([np.pi, np.pi, np.pi])
    obs_parser.gates_in_range = np.array([1, 1, 1])
    obs_parser.obstacles_pos = np.array([[1, 2, 3], [1, 1, 1]])
    obs_parser.obstacles_in_range = np.array([1, 1])
    obs_parser.gate_id = 1

    # This observation should NOT be in bounds
    if obs_parser.out_of_bounds():
        print("Out of bounds function is correct.")
    else:
        print(f"Out of bounds function is incorrect: {obs_parser.out_of_bounds()}")
        print("Expected: True")
        print(f"Observation space: {obs_parser.observation_space}")
        print(f"Observation: {obs_parser.get_observation()}")


if __name__ == "__main__":
    test_observation_parser()
