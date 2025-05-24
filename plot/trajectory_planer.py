import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from plot_new import plot_waypoints_and_environment
def optimize_drone_path(gates, gate_rotations, obstacles, max_curvature=0, num_points=50):
    """
    gates: list of 3D positions (N_gates x 3)
    gate_rotations: list of yaw angles in radians (N_gates)
    obstacles: list of (x, y, z, radius)
    max_curvature: maximum allowed curvature (1/m)
    num_points: number of points along the spline

    Returns: optimized path as (num_points x 3) array
    """
    N = num_points
    opti = ca.Opti()

    # Decision variables: path points in 3D
    X = opti.variable(N)
    Y = opti.variable(N)
    Z = opti.variable(N)

    # Objective: minimize total path length
    total_length = ca.sumsqr(ca.sqrt((X[1:] - X[:-1])**2 + (Y[1:] - Y[:-1])**2 + (Z[1:] - Z[:-1])**2))
    opti.minimize(total_length)

    # Pass through gates at evenly spaced indices
    gate_indices = np.linspace(0, N-1, len(gates)).astype(int)
    for idx, (gate, yaw) in zip(gate_indices, zip(gates, gate_rotations)):
        opti.subject_to(X[idx] == gate[0])
        opti.subject_to(Y[idx] == gate[1])
        opti.subject_to(Z[idx] == gate[2])

    # Obstacle avoidance constraints
    for ox, oy, oz in obstacles:
        for i in range(N):
            dist_squared = (X[i] - ox)**2 + (Y[i] - oy)**2
            opti.subject_to(dist_squared >= (10 + 0.05)**2)  # 5cm safety margin

    # Curvature constraints
    for i in range(1, N-1):
        dx1 = X[i] - X[i-1]
        dy1 = Y[i] - Y[i-1]
        dz1 = Z[i] - Z[i-1]
        dx2 = X[i+1] - X[i]
        dy2 = Y[i+1] - Y[i]
        dz2 = Z[i+1] - Z[i]

        v1 = ca.vertcat(dx1, dy1, dz1)
        v2 = ca.vertcat(dx2, dy2, dz2)
        norm1 = ca.norm_2(v1)
        norm2 = ca.norm_2(v2)
        angle = ca.acos(ca.dot(v1, v2) / (norm1 * norm2))

        curvature = angle / norm1
        opti.subject_to(curvature <= max_curvature)

    # Set start and end points
    opti.subject_to(X[0] == gates[0][0])
    opti.subject_to(Y[0] == gates[0][1])
    opti.subject_to(Z[0] == gates[0][2])
    opti.subject_to(X[-1] == gates[-1][0])
    opti.subject_to(Y[-1] == gates[-1][1])
    opti.subject_to(Z[-1] == gates[-1][2])

    # Solver options
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0})

    # Initial guess (linear interpolation)
    start = gates[0]
    end = gates[-1]

    opti.set_initial(X, np.linspace(start[0], end[0], N))
    opti.set_initial(Y, np.linspace(start[1], end[1], N))
    opti.set_initial(Z, np.linspace(start[2], end[2], N))
    # Solve optimization
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(e)
        print("Last values:")
        print("X:", opti.debug.value(X))
        print("Y:", opti.debug.value(Y))
        print("Z:", opti.debug.value(Z))
        plot_waypoints_and_environment([opti.debug.value(X),opti.debug.value(Y),opti.debug.value(Z)],obstacles,gates,gate_rotations)


    X_opt = sol.value(X)
    Y_opt = sol.value(Y)
    Z_opt = sol.value(Z)

    return np.vstack((X_opt, Y_opt, Z_opt)).T




obstacles_positions = [
    [0.6, 0.1, 0.6],
    [0.25, -0.65, 0.6],
    [0.75, -1.15, 1.1],
    [-0.3, 1.0, 0.55],
]

gates_positions = [
    [0.45, -0.5, 0.56],
    [1.0, -1.05, 1.11],
    [0.0, 1.0, 0.56],
    [-0.5, 0.0, 1.11],
]

gates_quat = [
    [0.0, 0.0, 0.92388, 0.38268],
    [0.0, 0.0, -0.38268, 0.92388],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
]

traj=optimize_drone_path(gates_positions,gates_quat,obstacles_positions)
print(traj)