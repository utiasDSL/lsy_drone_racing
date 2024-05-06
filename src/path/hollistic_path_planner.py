# check whether we have to add to path
import sys
import os
if not os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import time

from src.map.map import Map
from src.utils.calc_gate_center import calc_gate_center_and_normal
from src.path.rtt_star import RRTStar

if __name__ == "__main__":
    nomial_gates = np.array([[0.45, -1.0, 0, 0, 0, 2.35, 1], [1.0, -1.55, 0, 0, 0, -0.78, 0], [0.0, 0.5, 0, 0, 0, 0, 1], [-0.5, -0.5, 0, 0, 0, 3.14, 0]])
    nominal_obstacles = np.array([
    [1.0, -0.5, 0, 0, 0, 0],
      [0.5, -1.5, 0, 0, 0, 0],
      [-0.5, 0, 0, 0, 0, 0],
      [0, 1.0, 0, 0, 0, 0]
    ])
    gate_types = {'tall': {'shape': 'square', 'height': 1.0, 'edge': 0.45}, 'low': {'shape': 'square', 'height': 0.525, 'edge': 0.45}}
    start_point = np.array([1, 1, 0.05])
    goal_point = np.array([0, -2, 0.5])
    
    lower_bound = np.array([-1.5, -2, 0])
    upper_bound = np.array([1.5, 1, 1])
    drone_radius = 0.1

    map = Map(lower_bound=lower_bound, upper_bound=upper_bound, drone_radius=drone_radius)
    map.parse_gates(nomial_gates)
    map.parse_obstacles(nominal_obstacles)
    #map.easy_plot()

    gates_centers = []
    gates_normals = []
    for gate in nomial_gates:
        center, normal = calc_gate_center_and_normal(gate, gate_types)
        gates_centers.append(center)
        gates_normals.append(normal)
    
    # add the start and end points
    checkpoints = [start_point]
    for gate_center, gate_normal in zip(gates_centers, gates_normals):
        gate_normal_normalized = gate_normal / np.linalg.norm(gate_normal)
        # add checkpoint before and after the gate center, 10 cm
        eps = 0.05
        early_checkpoint = gate_center - (drone_radius + eps) * gate_normal_normalized 
        late_checkpoint = gate_center + (drone_radius + eps) * gate_normal_normalized
        checkpoints.append(early_checkpoint)
        #checkpoints.append(gate_center)
        checkpoints.append(late_checkpoint)

    checkpoints.append(goal_point)
    checkpoints = np.array(checkpoints)


    # Generate path using r_star
    start_time = time.time()
    path = []
    for i, (start_pos, end_pos) in enumerate(zip(checkpoints[:-1], checkpoints[1:])):
        print(f"Generating section {i} from {start_pos} to {end_pos}")
        
        rrt = RRTStar(start_pos, end_pos, map,  max_iter=500, max_extend_length=1, goal_sample_rate=0.1)
        waypoints, _ = rrt.plan()
        if waypoints is None:
            print("Failed to find path")
            exit(1)

        path.append(waypoints)

    
    path = np.concatenate(path, axis=0)
    # purge path remove consecutive points where difference is insignificant
    insignificance_threshold = 0.01
    purged_path = [path[0]]
    for point in path[1:]:
        if np.linalg.norm(purged_path[-1] - point) > insignificance_threshold:
            purged_path.append(point)

    purged_path = np.array(purged_path)
    print(f"Time taken: {time.time() - start_time} seconds")

    # visualize the path
    map.draw_scene(purged_path)

    # log path into file
    np.savetxt("path.txt", purged_path)
    