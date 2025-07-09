"""This module provides functionality to visualize drone racing trajectories with gates and obstacles in a 3D plot.

It processes trajectory data from NPZ files and renders them using matplotlib.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from scipy.spatial.transform import Rotation as R


def plot_trajectories(directory: str = "flight_logs", show_waypoints: bool = True):
    """Plot all trajectory NPZ files with gates and obstacles in 3D.

    Args:
        directory (str): Directory containing trajectory NPZ files
        show_waypoints (bool): Whether to plot waypoints as markers
    """
    # Find all NPZ files in the directory
    search_pattern = os.path.join(directory, "*.npz")
    trajectory_files = glob.glob(search_pattern)

    if not trajectory_files:
        print(f"No trajectory files found in {directory}")
        return

    print(f"Found {len(trajectory_files)} trajectory files")

    # Create figure for 3D plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Use colormap to distinguish between different files
    cmap = get_cmap("viridis")
    file_colors = cmap(np.linspace(0, 1, len(trajectory_files)))

    # Track legend entries
    legend_entries = []

    # Process each trajectory file
    for file_idx, trajectory_file in enumerate(trajectory_files):
        try:
            # Load trajectory data
            data = np.load(trajectory_file, allow_pickle=True)
            num_trajectories = int(data["num_trajectories"])

            file_basename = os.path.basename(trajectory_file)
            print(f"Processing {file_basename} with {num_trajectories} trajectories")

            # Plot each trajectory in this file
            for i in range(num_trajectories):
                traj_key = f"traj_{i}"
                if traj_key in data:
                    # Extract trajectory data
                    traj_data = data[traj_key].item()  # Convert from 0d array to dict

                    # Plot the main trajectory
                    tick = traj_data["tick"]
                    label = f"{file_basename} (tick {tick})"

                    # Use a gradient of the file color for each trajectory in the file
                    traj_color = file_colors[file_idx]
                    if num_trajectories > 1:
                        # Adjust alpha to distinguish trajectories from same file
                        alpha = 0.5 + 0.5 * (i / num_trajectories)
                    else:
                        alpha = 1.0

                    # Plot trajectory line
                    ax.plot(
                        traj_data["x"],
                        traj_data["y"],
                        traj_data["z"],
                        color=traj_color,
                        alpha=alpha,
                        linewidth=2,
                        label=label,
                    )
                    legend_entries.append(label)

                    # Plot waypoints if requested
                    if show_waypoints and "waypoints_x" in traj_data:
                        ax.scatter(
                            traj_data["waypoints_x"],
                            traj_data["waypoints_y"],
                            traj_data["waypoints_z"],
                            color=traj_color,
                            marker="o",
                            s=50,
                            alpha=0.7,
                        )

                        # Connect waypoints with dotted lines
                        ax.plot(
                            traj_data["waypoints_x"],
                            traj_data["waypoints_y"],
                            traj_data["waypoints_z"],
                            color=traj_color,
                            linestyle=":",
                            alpha=0.5,
                        )

        except Exception as e:
            print(f"Error processing {trajectory_file}: {e}")

    # Plot gates and obstacles from the provided data
    # Define gates
    gates = [
        {"pos": [0.45, -0.5, 0.56], "rpy": [0.0, 0.0, 2.35], "height": 0.525},
        {"pos": [1.0, -1.05, 1.11], "rpy": [0.0, 0.0, -0.78], "height": 1.0},
        {"pos": [0.0, 1.0, 0.56], "rpy": [0.0, 0.0, 0.0], "height": 0.525},
        {"pos": [-0.5, 0.0, 1.11], "rpy": [0.0, 0.0, 3.14], "height": 1.0},
    ]

    # Define obstacles
    obstacles = [[1.0, 0.0, 1.4], [0.5, -1.0, 1.4], [0.0, 1.5, 1.4], [-0.5, 0.5, 1.4]]

    # Plot gates as thick points with labels
    gate_x = [gate["pos"][0] for gate in gates]
    gate_y = [gate["pos"][1] for gate in gates]
    gate_z = [gate["pos"][2] for gate in gates]

    ax.scatter(
        gate_x,
        gate_y,
        gate_z,
        color="green",
        marker="s",  # square markers
        s=150,  # large size
        label="Gates",
        alpha=0.8,
    )

    # Add direction vectors for gates
    for i, gate in enumerate(gates):
        pos = gate["pos"]
        rpy = gate["rpy"]

        # Create rotation object
        rotation = R.from_euler("xyz", rpy)
        normal = rotation.apply([0.3, 0.0, 0.0])  # Scale the normal vector

        # Plot normal vector as arrow
        ax.quiver(pos[0], pos[1], pos[2], normal[0], normal[1], normal[2], color="green", alpha=0.7)

        # Add gate number label
        ax.text(pos[0], pos[1], pos[2] + 0.1, f"Gate {i + 1}", color="green")

    # Plot obstacles as thick points with labels
    obstacles_x = [obs[0] for obs in obstacles]
    obstacles_y = [obs[1] for obs in obstacles]
    obstacles_z = [obs[2] for obs in obstacles]

    ax.scatter(
        obstacles_x,
        obstacles_y,
        obstacles_z,
        color="red",
        marker="o",  # circle markers
        s=150,  # large size
        label="Obstacles",
        alpha=0.8,
    )

    # Add obstacle labels
    for i, obs in enumerate(obstacles):
        ax.text(obs[0], obs[1], obs[2] + 0.1, f"Obs {i + 1}", color="red")

    # Add gate and obstacle entries to legend
    legend_entries = ["Gates", "Obstacles"] + legend_entries

    # Set plot labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Drone Racing Trajectories with Gates and Obstacles")

    # Add a smaller legend to avoid overcrowding
    if len(legend_entries) > 12:
        # If too many entries, limit the legend
        legend_entries = legend_entries[:12] + ["..."]
    ax.legend(legend_entries, loc="upper right", fontsize="small")

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 0.5])  # Make z axis half scale for better visualization

    # Add grid
    ax.grid(True, alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # You can specify a different directory if needed
    plot_trajectories()
