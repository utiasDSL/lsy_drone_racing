import numpy as np
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R


def load_track(track: ConfigDict) -> tuple[ConfigDict, ConfigDict, ConfigDict]:
    """Load the track from the config file."""
    assert "gates" in track, "Track must contain gates field."
    assert "obstacles" in track, "Track must contain obstacles field."
    assert "drone" in track, "Track must contain drone field."
    gate_pos = np.array([g["pos"] for g in track.gates], dtype=np.float32)
    gate_quat = (
        R.from_euler("xyz", np.array([g["rpy"] for g in track.gates])).as_quat().astype(np.float32)
    )
    gates = {"pos": gate_pos, "quat": gate_quat, "nominal_pos": gate_pos, "nominal_quat": gate_quat}
    obstacle_pos = np.array([o["pos"] for o in track.obstacles], dtype=np.float32)
    obstacles = {"pos": obstacle_pos, "nominal_pos": obstacle_pos}
    drone_keys = ("pos", "rpy", "vel", "ang_vel")
    drone = {k: np.array(track.drone.get(k), dtype=np.float32) for k in drone_keys}
    drone["quat"] = R.from_euler("xyz", drone["rpy"]).as_quat().astype(np.float32)
    return ConfigDict(gates), ConfigDict(obstacles), ConfigDict(drone)
