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

waypoints=[[ 1.,1.5 ,0.05],
 [ 0.97807474 , 1.41906259,  0.07093467],
 [ 0.95722695 , 1.35856363,  0.08801094],
 [ 0.93718154  ,1.31328481 , 0.10221392],
 [ 0.91766339,  1.27800779 , 0.11452874],
 [ 0.89839742 , 1.24751423 , 0.12594053],
 [ 0.87910853 , 1.21658582 , 0.13743441],
 [ 0.85952161 , 1.18000422 , 0.14999552],
 [ 0.83936157 , 1.13255109 , 0.16460897],
 [ 0.8183533  , 1.06900811 , 0.18225989],
 [ 0.79622202 , 0.98416553 , 0.20393177],
 [ 0.77279455 , 0.87572152 , 0.23005236],
 [ 0.74814685 , 0.74850259 , 0.25968711],
 [ 0.72239237 , 0.60840747 , 0.29169658],
 [ 0.69564453 , 0.46133491 , 0.32494129],
 [ 0.6680193  , 0.31317301 , 0.3582842 ],
 [ 0.63993829 , 0.16852348 , 0.39088005],
 [ 0.6124167  , 0.02948965 , 0.42245028],
 [ 0.58653794 ,-0.1021122  , 0.45278141],
 [ 0.56338542 ,-0.22446579 , 0.48165999],
 [ 0.54399283 ,-0.33580057 , 0.50886282],
 [ 0.52641067 ,-0.43708961 , 0.53358171],
 [ 0.50406552 ,-0.53355861 , 0.55410179],
 [ 0.46998619 ,-0.6307991  , 0.56863018],
 [ 0.41720151 ,-0.7344026  , 0.57537402],
 [ 0.33917788 ,-0.84973675 , 0.57269671],
 [ 0.24483091 ,-0.97426555 , 0.56447965],
 [ 0.16228524 ,-1.09562587 , 0.56146507],
 [ 0.12086633 ,-1.20084025 , 0.5748241 ],
 [ 0.1498996  ,-1.27693124 , 0.61572786],
 [ 0.27693158 ,-1.31115757 , 0.69461558],
 [ 0.48859395 ,-1.29621059 , 0.80509241],
 [ 0.73060347 ,-1.23021423 , 0.92392951],
 [ 0.94689802 ,-1.11152864 , 1.0271661 ],
 [ 1.08141548 ,-0.93851396 , 1.09084139],
 [ 1.08159545 ,-0.7103144  , 1.09245912],
 [ 0.95089526 ,-0.43861729 , 1.03295063],
 [ 0.73782527 ,-0.14519788 , 0.93208927],
 [ 0.49217201 , 0.1478828  , 0.81018212],
 [ 0.26372199 , 0.41856375 , 0.68753625],
 [ 0.09896964 , 0.64628606 , 0.58368175],
 [ 0.00613888 , 0.82795281 , 0.50911636],
 [-0.03123703 , 0.97173292 , 0.46851048],
 [-0.03003631 , 1.08598304 , 0.46643743],
 [-0.00713722 , 1.17905984 , 0.50747048],
 [ 0.02124037 , 1.25775577 , 0.59447836],
 [ 0.04460691 , 1.31524897 , 0.7154938 ],
 [ 0.05542331 , 1.33770761 , 0.85091054],
 [ 0.0461749  , 1.31124192 , 0.9810592 ],
 [ 0.00934698 , 1.22196217 , 1.08627038],
 [-0.06073156 , 1.05943643 , 1.14971121],
 [-0.15747583 , 0.83622058 , 1.17340576],
 [-0.26930119 , 0.57424818 , 1.16707068],
 [-0.38460824 , 0.29548043  ,1.14044531],
 [-0.49179758 , 0.02187857 , 1.10326899],
 [-0.57926983 ,-0.2245962 ,  1.06528106],
 [-0.63542559 ,-0.42198266 , 1.03622086],
 [-0.64866546 ,-0.54831959,  1.02582772],
 [-0.60739007 ,-0.58164578 , 1.04384099],
 [-0.5        ,-0.5        , 1.1       ],]

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

gates_quat = [
    [0.0, 0.0, 0.92388, 0.38268],
    [0.0, 0.0, -0.38268, 0.92388],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
]

plot_waypoints_and_environment(waypoints, obstacles_positions, gates_positions, gates_quat)
