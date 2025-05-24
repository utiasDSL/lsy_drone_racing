import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def plot_waypoints_and_environment(waypoints, obstacle_positions, gates_positions, gates_quat):
    def quaternion_to_rotation_matrix(q):
        x, y, z, w = q
        R = np.array([
            [1-2*(y**2+z**2),  2*(x*y - z*w),    2*(x*z + y*w)],
            [2*(x*y + z*w),    1-2*(x**2+z**2),  2*(y*z - x*w)],
            [2*(x*z - y*w),    2*(y*z + x*w),    1-2*(x**2+y**2)]
        ])
        return R

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Waypoints plotten
    waypoints = np.array(waypoints)
    ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], 'bo-', label='Waypoints', markersize=2)

    # Gates als rotierte Quadrate
    gate_size = 0.2
    gate_color = (1, 0, 0, 0.5)
    gates_positions = np.array(gates_positions)

    for i, gate in enumerate(gates_positions):
        quat = gates_quat[i]
        R = quaternion_to_rotation_matrix(quat)

        local_square = np.array([
            [-gate_size, 0, -gate_size],
            [ gate_size, 0, -gate_size],
            [ gate_size, 0,  gate_size],
            [-gate_size, 0,  gate_size],
        ])

        rotated_square = (R @ local_square.T).T
        square = rotated_square + gate

        poly = Poly3DCollection([square], color=gate_color, label='Gate' if i == 0 else "")
        ax.add_collection3d(poly)

    # Gate-Zentren markieren
    ax.scatter(gates_positions[:,0], gates_positions[:,1], gates_positions[:,2], c='r', s=50, label=None)

    # Stäbe plotten (von z=0 bis z=1)
    for idx, point in enumerate(obstacle_positions):
        ax.plot([point[0], point[0]], [point[1], point[1]], [0, 1],
                linewidth=4, label='Staves' if idx == 0 else "")

    # Achsenbeschriftung und Limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)

    ax.legend()
    ax.set_title('3D Waypoints mit Stäben und rotierbaren Gates')
    plt.show()
'''
waypoints = [
    [1.0, 1.5, 0.2],
    [0.9, 1.25, 0.2],
    [0.8, 1.0, 0.2],
    [0.625, 0.25, 0.38],
    [0.45, -0.5, 0.56],
    [0.325, -0.9, 0.605],
    [0.2, -1.3, 0.65],
    [0.6, -1.175, 0.88],
    [1.0, -1.05, 1.11],
    [0.6, -0.275, 0.88],
    [0.2, 0.5, 0.65],
    [0.1, 0.75, 0.605],
    [0.0, 1.0, 0.56],
    [0.0, 1.1, 0.83],
    [0.0, 1.2, 1.1],
    [-0.25, 0.6, 1.105],
    [-0.5, 0.0, 1.11],
    [-0.5, -0.25, 1.105],
    [-0.5, -0.5, 1.1],
]

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

plot_waypoints_and_environment(waypoints, obstacles_positions, gates_positions, gates_quat)
'''