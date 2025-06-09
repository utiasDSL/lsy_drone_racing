import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.interpolate import CubicSpline



def plot_waypoints_and_environment(waypoints, obstacle_positions, gates_positions, gates_quat,show_spline=True):
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

    waypoints = np.array(waypoints)
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'bo-', label='Waypoints', markersize=2)

    # Optional: cubic spline Trajektorie
    if show_spline:
        ts = np.linspace(0, 1, len(waypoints))
        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])

        t_fine = np.linspace(0, 1, 200)
        spline_x = cs_x(t_fine)
        spline_y = cs_y(t_fine)
        spline_z = cs_z(t_fine)

        ax.plot(spline_x, spline_y, spline_z, 'g-', linewidth=2, label='Cubic Spline Trajectory')

    # Gates as rotated squares
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

    ax.scatter(gates_positions[:, 0], gates_positions[:, 1], gates_positions[:, 2], c='r', s=50, label=None)

    # Staves
    for idx, point in enumerate(obstacle_positions):
        ax.plot([point[0], point[0]], [point[1], point[1]], [0, 1],
                linewidth=4, label='Staves' if idx == 0 else "")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set axis limits to nearest multiples of 0.15 that cover the data
    def get_grid_limits(data, step=0.15):
        dmin = np.min(data)
        dmax = np.max(data)
        lower = step * np.floor(dmin / step)
        upper = step * np.ceil(dmax / step)
        return lower, upper

    xlim = get_grid_limits(np.concatenate([waypoints[:,0], np.array(gates_positions)[:,0], np.array(obstacle_positions)[:,0]]))
    ylim = get_grid_limits(np.concatenate([waypoints[:,1], np.array(gates_positions)[:,1], np.array(obstacle_positions)[:,1]]))
    zlim = get_grid_limits(np.concatenate([waypoints[:,2], np.array(gates_positions)[:,2], np.array(obstacle_positions)[:,2]]))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # Set grid lines every 0.15
    ax.set_xticks(np.arange(xlim[0], xlim[1]+0.001, 0.15))
    ax.set_yticks(np.arange(ylim[0], ylim[1]+0.001, 0.15))
    ax.set_zticks(np.arange(zlim[0], zlim[1]+0.001, 0.15))

    ax.legend()
    ax.set_title('3D Waypoints mit Stäben, rotierbaren Gates' +
                 (' und Spline-Trajektorie' if show_spline else ''))
    plt.show()

waypoints = np.array(
            [
            [1.0, 1.5, 0.2],
            [0.625, 0.25, 0.38],
            [0.45, -0.5, 0.56],
            [0.425, -0.57, 0.56],
            [0.325, -0.9, 0.605],
            [0.2, -1.3, 0.65],
            [0.6, -1.375, 0.78],
            [0.8, -1.375, 0.88],
            [1.0, -1.05, 1.11], # [1.0, -1.05, 1.11],
            [1.1, -0.8, 1.11],
            [0.7, -0.275, 0.88],
            [0.2, 0.5, 0.65],
            [0.0, 1.0, 0.56],
            [0.0, 1.05, 0.56],
            [0.0, 0.9, 0.63],
            [-0.1, 0.7, 0.75],
            [-0.25, 0.3, 0.95],
            #[-0.5, 0.0, 1.1],
            [-0.42, -0.4, 1.11],
            ])


obstacles_positions = [
    [1, -0.0, 1.4],
    [0.5, -1.0, 1.4],
    [0.0, 1.5, 1.4],
    [-0.5, 0.5, 1.4],
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