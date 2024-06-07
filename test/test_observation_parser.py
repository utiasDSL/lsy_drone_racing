"""Script to test the observation parser module."""
import numpy as np

from lsy_drone_racing.env_modifiers import ObservationParser, make_observation_parser


def test_observation_parser():
    """Test the observation parser."""
    n_gates = 3
    n_obstacles = 2
    obs_parser = make_observation_parser(n_gates, n_obstacles, observation_parser_type="classic")

    # Check the uninitialized state
    assert obs_parser.uninitialized()
    if obs_parser.uninitialized():
        print("Uninitialized function is correct.")
    else:
        raise ValueError("Uninitialized function is incorrect.")

    # Check the observable variables
    obs_parser.drone_pos = np.array([1, 2, 3])
    obs_parser.drone_rpy = np.array([0, 0, 0])
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

    n_gates = 4
    n_obstacles = 0

    obs_parser = make_observation_parser(n_gates, n_obstacles, observation_parser_type="scaramuzza") 

    obs_parser.drone_pos = np.array([1, 1, 1])
    obs_parser.drone_rpy = np.array([0, 0, 0])
    obs_parser.gates_pos = np.array([[1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1]])
    obs_parser.gates_yaw = np.array([3*np.pi/4, np.pi/4, 3*np.pi/4, np.pi/4])
    obs_parser.gate_edge_size = 2*np.sqrt(2)
    obs_parser.gate_id = 0
    obs_parser.reference_position = np.array([1, 1, 1])

    corners=obs_parser.get_relative_corners(include_reference_position=True)
    print(corners.shape)
    print(corners[:,0,:])
    print(corners[:,1,:])
    print(corners[:,2,:])
    gate_id = 2
    gates_in_sight = range(gate_id, gate_id + 4)
    gates_in_sight = [i if i < n_gates else -1 for i in gates_in_sight]
    print(gates_in_sight)

    corners_gates_in_sight = np.array([corners[:,i,:] for i in gates_in_sight])
    print(corners_gates_in_sight.shape)
    print(corners_gates_in_sight)


if __name__ == "__main__":
    test_observation_parser()
