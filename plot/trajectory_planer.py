import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from plot_new import plot_waypoints_and_environment

def optimize_trajectory(x0, obstacle, gate,gate_quat,initial_waypoints, n_waypoints=9, obstacle_radius=0.25):
    """
    Optimize drone trajectory from start point through gate while avoiding obstacle.
    
    Args:
        x0: Starting position [x, y, z] (first waypoint is fixed to this)
        obstacle: Obstacle position [x, y, z] (z ignored - treated as vertical pole)
        gate: Gate position [x, y, z] (last waypoint must reach this)
        n_waypoints: Number of waypoints in trajectory
        obstacle_radius: Safety radius around obstacle
        
    Returns:
        List of 3D waypoints [[x,y,z], ...] representing optimal trajectory
    """
    # Transform everything in np.array()
    x0 = np.array(x0)
    gate = np.array(gate)
    obstacle = np.array(obstacle[:2])
    initial_waypoints=np.array(initial_waypoints)
    gate_quat=np.array(gate_quat)




    R_gate = R.from_quat(gate_quat).as_matrix()
    gate_dir = R_gate[:, 1]  # Assuming Y-axis is forward
    fly_through = gate_dir * 0.1
    
    # Create initial guess (straight line from x0 to gate)
    #initial_waypoints = np.linspace(x0, gate, n_waypoints)
    
    # The first waypoint is fixed to x0, so we only optimize waypoints[1:]
    initial_guess = initial_waypoints[1:].flatten()
    
    # Define optimization problem

    
    def objective(x):
        """Minimize deviation from initial guess and evenly space points"""
        # Reconstruct full trajectory (x0 + optimized waypoints)
        optimized_waypoints = np.vstack([x0, x.reshape(-1, 3)])
        
        # 1. Minimize total path length
        distances = np.linalg.norm(np.diff(optimized_waypoints, axis=0), axis=1)
        path_length = np.sum(distances)
        
        # 2. Penalize deviation from initial guess (excluding x0)
        deviation = np.linalg.norm(optimized_waypoints[1:] - initial_waypoints[1:], axis=1)
        deviation_penalty = np.sum(deviation**2)  # Squared L2 norm
        avg_distance = path_length / (n_waypoints-1)
        spacing_penalty = np.sum((distances - avg_distance)**2)

        return  100*deviation_penalty+ spacing_penalty
    
    
    def constraint_gate(x):
        """
        Constraints:
        - Second-last waypoint must be at gate (equality).
        - Last waypoint must be behind gate along its orientation (equality).
        """
        optimized_waypoints = np.vstack([x0, x.reshape(-1, 3)])
        
        gate_constraint = optimized_waypoints[-2] - gate_pos
        
        # Last waypoint == gate + gate_dir * exit_length (3 equality constraints)
        exit_constraint = optimized_waypoints[-1] - (gate_pos-fly_through)
        
        return np.concatenate([gate_constraint, exit_constraint])
    
    def constraint_obstacle(x):
        """All waypoints must maintain safe distance from obstacle"""
        waypoints = np.vstack([x0, x.reshape(-1, 3)])
        xy_distances = np.linalg.norm(waypoints[:, :2] - obstacle, axis=1)
        return xy_distances - obstacle_radius
    
    def constraint_progress(x):
        """Waypoints should generally progress toward gate"""
        waypoints = np.vstack([x0, x.reshape(-1, 3)])
        progress = []
        for i in range(1, len(waypoints)):
            # Distance to gate should be non-increasing
            prev_dist = np.linalg.norm(waypoints[i-1] - gate)
            curr_dist = np.linalg.norm(waypoints[i] - gate)
            progress.append(prev_dist - curr_dist)
        return np.array(progress)
    
    constraints = [
        {'type': 'eq', 'fun': constraint_gate},  # Must reach gate
        {'type': 'ineq', 'fun': constraint_obstacle}#,  # Must avoid obstacle
        #{'type': 'ineq', 'fun': constraint_progress}  # Must make progress
    ]
    
    # Solve optimization
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True}
    )
    
    if not result.success:
        print("Optimization warning:", result.message)
    
    # Reconstruct full trajectory including fixed first waypoint
    optimized_waypoints = np.vstack([x0, result.x.reshape(-1, 3)])
    
    return optimized_waypoints.tolist()

# Test case Gate 1->2
waypoints = [
    [0.45, -0.5, 0.56],
    [(0.45 + 0.325)/2, (-0.5 + -0.9)/2, (0.56 + 0.605)/2],
    [0.325, -0.9, 0.605],
    [(0.325 + 0.5)/2, (-0.9 + -1)/2, (0.605 + 0.65)/2],
    [0.5, -1, 0.65],
    [(0.5 + 0.6)/2, (-1 + -1.175)/2, (0.65 + 0.88)/2],
    [0.6, -1.175, 0.88],
    [(0.6 + 1.0)/2, (-1.175 + -1.05)/2, (0.88 + 1.11)/2],
    [1.0, -1.05, 1.11],
]
x_0 = [0.45, -0.5, 0.56]
obstacle_pos = [0.5, -1, 1.4]  
gate_pos = [1.0, -1.05, 1.11]
gates_quat = [0.0, 0.0, 0.92388, 0.38268]
# Test case Gate 2->3




trajectory = optimize_trajectory(x_0, obstacle_pos, gate_pos,gates_quat,waypoints)
print("Optimized trajectory:")
for i, point in enumerate(trajectory):
    print(f"Waypoint {i}: {point}")

gates_quat = [
    [0.0, 0.0, 0.92388, 0.38268],
    [0.0, 0.0, -0.38268, 0.92388],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
]
obstacles_positions = [
    [1, 0, 1.4],
    [0.5, -1, 1.4],
    [0, 1.5, 1.4],
    [-0.5, 0.5, 1.4],
]

gates_positions = [
    [0.45, -0.5, 0.56],
    [1.0, -1.05, 1.11],
    [0.0, 1.0, 0.56],
    [-0.5, 0.0, 1.11],
]

plot_waypoints_and_environment(trajectory, obstacles_positions, gates_positions, gates_quat)