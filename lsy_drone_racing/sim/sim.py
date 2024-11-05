"""Quadrotor simulation environment using PyBullet physics engine.

This module implements a simulation environment for quadrotor drones using PyBullet. It provides
functionality for simulating drone dynamics, control, and environmental interactions.

Features:

* PyBullet-based physics simulation
* Configurable drone parameters and initial conditions
* Support for a single drone (multi-drone support not yet implemented)
* Disturbance and randomization options
* Integration with symbolic models

The simulation is derived from the gym-pybullet-drones project:
https://github.com/utiasDSL/gym-pybullet-drones

This environment can be used for developing and testing drone control algorithms, path planning
strategies, and other robotics applications in a simulated 3D space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from gymnasium.utils import seeding

from lsy_drone_racing.sim.drone import Drone
from lsy_drone_racing.sim.noise import NoiseList
from lsy_drone_racing.sim.physics import (
    GRAVITY,
    PhysicsMode,
    apply_force_torques,
    force_torques,
    pybullet_step,
    sys_id_dynamics,
)
from lsy_drone_racing.sim.symbolic import SymbolicModel, symbolic
from lsy_drone_racing.utils import map2pi

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Sim:
    """Drone simulation based on pybullet and models from the Learning Systems and Robotics Lab.

    We implement a simplified version of the gym-pybullet-drones environment specifically designed
    for drone racing.
    """

    URDF_DIR = Path(__file__).resolve().parent / "assets"

    def __init__(
        self,
        track: dict,
        sim_freq: int = 500,
        ctrl_freq: int = 500,
        disturbances: dict | None = None,
        randomization: dict | None = None,
        gui: bool = False,
        camera_view: tuple[float, ...] = (5.0, -40.0, -40.0, 0.5, -1.0, 0.5),
        n_drones: int = 1,
        physics: PhysicsMode = PhysicsMode.PYB,
    ):
        """Initialization method for BenchmarkEnv.

        Args:
            track: The configuration of gates and obstacles. Must contain at least the initial drone
                state and can contain gates and obstacles.
            sim_freq: The frequency at which PyBullet steps (a multiple of ctrl_freq).
            ctrl_freq: The frequency at which the onboard drone controller recalculates the rotor
                rmps.
            disturbances: Dictionary to specify disturbances being used.
            randomization: Dictionary to specify randomization of the environment.
            gui: Option to show PyBullet's GUI.
            camera_view: The camera pose for the GUI.
            n_drones: The number of drones in the simulation. Only supports 1 at the moment.
            physics: The physics backend to use for the simulation. For more information, see the
                PhysicsMode enum.
        """
        self.np_random = np.random.default_rng()
        assert n_drones == 1, "Only one drone is supported at the moment."
        self.drone = Drone(controller="mellinger")
        self.n_drones = n_drones
        self.pyb_client = p.connect(p.GUI if gui else p.DIRECT)
        self.settings = SimSettings(sim_freq, ctrl_freq, gui, pybullet_id=self.pyb_client)
        self.physics_mode = PhysicsMode(physics)

        # Create the state and action spaces of the simulation. Note that the state space is
        # different from the observation space of any derived environment.
        min_thrust, max_thrust = self.drone.params.min_thrust, self.drone.params.max_thrust
        self.action_space = spaces.Box(low=min_thrust, high=max_thrust, shape=(4,))
        # pos in meters, rpy in radians, vel in m/s ang_vel in rad/s
        rpy_max = np.array([85 / 180 * np.pi, 85 / 180 * np.pi, np.pi], np.float32)  # Yaw unbounded
        pos_low, pos_high = np.array([-3, -3, 0]), np.array([3, 3, 2.5])
        # State space uses 64-bit floats for better compatibility with pycffirmware.
        self.state_space = spaces.Dict(
            {
                "pos": spaces.Box(low=pos_low, high=pos_high, dtype=np.float64),
                "rpy": spaces.Box(low=-rpy_max, high=rpy_max, dtype=np.float64),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            }
        )
        self.disturbances = self._setup_disturbances(disturbances)
        self.randomization = {} if randomization is None else randomization
        if self.settings.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=camera_view[0],
                cameraYaw=camera_view[1],
                cameraPitch=camera_view[2],
                cameraTargetPosition=[camera_view[3], camera_view[4], camera_view[5]],
                physicsClientId=self.pyb_client,
            )

        assert isinstance(track.drone, dict), "Expected drone state as dictionary."
        for key, val in track.drone.items():
            assert hasattr(self.drone, "init_" + key), f"Unknown key '{key}' in drone init state."
            setattr(self.drone, "init_" + key, np.array(val, dtype=float))

        self.pyb_objects = {}  # Populated when objects are loaded into PyBullet.

        self.gates = {}
        if gates := track.get("gates"):
            self.gates = {i: g.toDict() for i, g in enumerate(gates)}
            # Add nominal values to not loose the default when randomizing.
            for i, gate in self.gates.items():
                self.gates[i].update({"nominal." + k: v for k, v in gate.items()})
        self.n_gates = len(self.gates)

        self.obstacles = {}
        if obstacles := track.get("obstacles"):
            self.obstacles = {i: o.toDict() for i, o in enumerate(obstacles)}
            for i, obstacle in self.obstacles.items():
                self.obstacles[i].update({"nominal." + k: v for k, v in obstacle.items()})
        self.n_obstacles = len(self.obstacles)

        self._setup_pybullet()

    def step(self, desired_thrust: NDArray[np.floating]):
        """Advance the environment by one control step.

        Args:
            desired_thrust: The desired thrust for the drone.
        """
        self.drone.desired_thrust[:] = desired_thrust
        rpm = self._thrust_to_rpm(desired_thrust)  # Pre-process/clip the action
        disturb_force = np.zeros(3)
        if "dynamics" in self.disturbances:
            disturb_force = self.disturbances["dynamics"].apply(disturb_force)
        for _ in range(self.settings.sim_freq // self.settings.ctrl_freq):
            self.drone.rpm[:] = rpm  # Save the last applied action (e.g. to compute drag)
            dt = 1 / self.settings.sim_freq
            ft = force_torques(self.drone, rpm, self.physics_mode, dt, self.pyb_client)
            apply_force_torques(self.pyb_client, self.drone, ft, disturb_force)
            pybullet_step(self.pyb_client, self.drone, self.physics_mode)
            self._sync_pyb_to_sim()

    def step_sys_id(self, collective_thrust: float, rpy: NDArray[np.floating], dt: float):
        """Step the simulation with a system identification dynamics model.

        The signature of this function is different from step(), since we do not pass desired
        thrusts for each rotor, but instead a single collective thrust and attitude command.

        Note:
            This function does not simulate the onboard controller and instead directly sets the new
            drone state based on the control inputs.

        Warning:
            This is an experimental feature and is likely subject to change.

        Todo:
            The deviation of step_sys_id() from step() is not ideal. We should aim for a unified
            step function for all physics modes in the future.
        """
        sys_id_dynamics(self.drone, collective_thrust, rpy, dt)
        pybullet_step(self.pyb_client, self.drone, self.physics_mode)
        self._sync_pyb_to_sim()

    def reset(self):
        """Reset the simulation to its original state."""
        for mode in self.disturbances.keys():
            self.disturbances[mode].reset()
        self._randomize_gates()
        self._randomize_obstacles()
        self._randomize_drone()
        self._sync_pyb_to_sim()
        self.drone.reset()

    @property
    def collisions(self) -> list[int]:
        """Return the pybullet object IDs of the objects currently in collision with the drone."""
        collisions = []
        for o_id in self.pyb_objects.values():
            if p.getContactPoints(bodyA=o_id, bodyB=self.drone.id, physicsClientId=self.pyb_client):
                collisions.append(o_id)
        return collisions

    def in_range(self, bodies: dict, target_body: Drone, distance: float) -> NDArray[np.bool_]:
        """Return a mask array of objects within a certain distance of the drone."""
        in_range = np.zeros(len(bodies), dtype=bool)
        for i, body in enumerate(bodies.values()):
            assert "id" in body, "Body must have a PyBullet ID."
            closest_points = p.getClosestPoints(
                bodyA=body["id"],
                bodyB=target_body.id,
                distance=distance,
                physicsClientId=self.pyb_client,
            )
            in_range[i] = len(closest_points) > 0
        return in_range

    def seed(self, seed: int | None = None) -> int | None:
        """Set up a random number generator for a given seed."""
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        for noise in self.disturbances.values():
            noise.seed(seed)
        return seed

    def render(self) -> NDArray[np.uint8]:
        """Retrieve a frame from PyBullet rendering.

        Returns:
            The RGB frame captured by PyBullet's camera as [h, w, 4] tensor.
        """
        [w, h, rgb, _, _] = p.getCameraImage(
            width=self.settings.render_resolution[0],
            height=self.settings.render_resolution[1],
            shadow=1,
            viewMatrix=self.settings.camera_view,
            projectionMatrix=self.settings.camera_projection,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self.pyb_client,
        )
        return np.reshape(rgb, (h, w, 4))

    def close(self):
        """Stop logging and disconnect from the PyBullet simulation."""
        if self.pyb_client >= 0:
            p.disconnect(physicsClientId=self.pyb_client)
        self.pyb_client = -1

    def _setup_pybullet(self):
        """Setup the PyBullet simulation environment."""
        # Set up the simulation parameters.
        p.resetSimulation(physicsClientId=self.pyb_client)
        p.setGravity(0, 0, -GRAVITY, physicsClientId=self.pyb_client)
        p.setRealTimeSimulation(0, physicsClientId=self.pyb_client)
        p.setTimeStep(1 / self.settings.sim_freq, physicsClientId=self.pyb_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.pyb_client)
        # Load the ground plane, drone, gates and obstacles models.
        plane_id = p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self.pyb_client)
        self.pyb_objects["plane"] = plane_id
        self.drone.id = p.loadURDF(
            str(self.URDF_DIR / "cf2x.urdf"),
            self.drone.init_pos,
            p.getQuaternionFromEuler(self.drone.init_rpy),
            flags=p.URDF_USE_INERTIA_FROM_FILE,  # Use URDF inertia tensor.
            physicsClientId=self.pyb_client,
        )
        # Remove default damping.
        p.changeDynamics(self.drone.id, -1, linearDamping=0, angularDamping=0)

        # Load the gates.
        for i, gate in self.gates.items():
            self.gates[i]["pos"] = np.array(gate["nominal.pos"])
            self.gates[i]["id"] = self._load_urdf_into_sim(
                self.URDF_DIR / "gate.urdf",
                self.gates[i]["pos"],
                self.gates[i]["rpy"],
                marker=str(i),
            )
            self.pyb_objects[f"gate_{i}"] = self.gates[i]["id"]

        # Load the obstacles.
        for i, obstacle in self.obstacles.items():
            self.obstacles[i]["pos"] = np.array(obstacle["nominal.pos"])
            self.obstacles[i]["id"] = self._load_urdf_into_sim(
                self.URDF_DIR / "obstacle.urdf", self.obstacles[i]["pos"], marker=str(i)
            )
            self.pyb_objects[f"obstacle_{i}"] = self.obstacles[i]["id"]

        self._sync_pyb_to_sim()

    def _setup_disturbances(self, disturbances: dict | None = None) -> dict[str, NoiseList]:
        """Creates attributes and action spaces for the disturbances.

        Args:
            disturbances: A dictionary of disturbance configurations for the environment.

        Returns:
            A dictionary of NoiseList that fuse disturbances for each mode.
        """
        dist = {}
        if disturbances is None:  # Default: no passive disturbances.
            return dist
        modes = {"action": {"dim": spaces.flatdim(self.action_space)}, "dynamics": {"dim": 3}}
        for mode, spec in disturbances.items():
            assert mode in modes, "Disturbance mode not available."
            spec["dim"] = modes[mode]["dim"]
            dist[mode] = NoiseList.from_specs([spec])
        return dist

    def _randomize_obstacles(self):
        """Randomize the obstacles' position."""
        for obstacle in self.obstacles.values():
            pos_offset = np.zeros(3)
            if obstacle_pos := self.randomization.get("obstacle_pos"):
                distrib = getattr(self.np_random, obstacle_pos.get("type"))
                kwargs = {k: v for k, v in obstacle_pos.items() if k != "type"}
                pos_offset = distrib(**kwargs)
            obstacle["pos"] = np.array(obstacle["nominal.pos"]) + pos_offset
            p.resetBasePositionAndOrientation(
                obstacle["id"],
                obstacle["pos"],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.pyb_client,
            )

    def _randomize_gates(self):
        """Randomize the gates' position and orientation."""
        for gate in self.gates.values():
            pos_offset = np.zeros_like(gate["nominal.pos"])
            if gate_pos := self.randomization.get("gate_pos"):
                distrib = getattr(self.np_random, gate_pos.get("type"))
                pos_offset = distrib(**{k: v for k, v in gate_pos.items() if k != "type"})
            rpy_offset = np.zeros(3)
            if gate_rpy := self.randomization.get("gate_rpy"):
                distrib = getattr(self.np_random, gate_rpy.get("type"))
                rpy_offset = distrib(**{k: v for k, v in gate_rpy.items() if k != "type"})
            gate["pos"] = np.array(gate["nominal.pos"]) + pos_offset
            gate["rpy"] = map2pi(np.array(gate["nominal.rpy"]) + rpy_offset)  # Ensure [-pi, pi]
            p.resetBasePositionAndOrientation(
                gate["id"],
                gate["pos"],
                p.getQuaternionFromEuler(gate["rpy"]),
                physicsClientId=self.pyb_client,
            )

    def _load_urdf_into_sim(
        self,
        urdf_path: Path,
        pos: NDArray[np.floating],
        rpy: NDArray[np.floating] | None = None,
        marker: str | None = None,
    ) -> int:
        """Load a URDF file into the simulation.

        Args:
            urdf_path: Path to the URDF file.
            pos: Position of the object in the simulation.
            rpy: Roll, pitch, yaw orientation of the object in the simulation.
            marker: Optional text marker to display above the object

        Returns:
            int: The ID of the object in the simulation.
        """
        quat = p.getQuaternionFromEuler(rpy if rpy is not None else np.zeros(3))
        pyb_id = p.loadURDF(str(urdf_path), pos, quat, physicsClientId=self.pyb_client)
        if marker is not None:
            p.addUserDebugText(
                str(marker),
                textPosition=[0, 0, 0.5],
                textColorRGB=[1, 0, 0],
                textSize=1.5,
                parentObjectUniqueId=pyb_id,
                parentLinkIndex=-1,
                physicsClientId=self.pyb_client,
            )
        return pyb_id

    def _randomize_drone(self):
        """Randomize the drone's position, orientation and physical properties."""
        inertia_diag = self.drone.nominal_params.J.diagonal()
        if drone_inertia := self.randomization.get("drone_inertia"):
            distrib = getattr(self.np_random, drone_inertia.type)
            kwargs = {k: v for k, v in drone_inertia.items() if k != "type"}
            inertia_diag = inertia_diag + distrib(**kwargs)
            assert all(inertia_diag > 0), "Negative randomized inertial properties."
        self.drone.params.J = np.diag(inertia_diag)

        mass = self.drone.nominal_params.mass
        if drone_mass := self.randomization.get("drone_mass"):
            distrib = getattr(self.np_random, drone_mass.type)
            mass += distrib(**{k: v for k, v in drone_mass.items() if k != "type"})
            assert mass > 0, "Negative randomized mass."
        self.drone.params.mass = mass

        p.changeDynamics(
            self.drone.id,
            linkIndex=-1,  # Base link.
            mass=mass,
            localInertiaDiagonal=inertia_diag,
            physicsClientId=self.pyb_client,
        )

        pos = self.drone.init_pos.copy()
        if drone_pos := self.randomization.get("drone_pos"):
            distrib = getattr(self.np_random, drone_pos.type)
            pos += distrib(**{k: v for k, v in drone_pos.items() if k != "type"})

        rpy = self.drone.init_rpy.copy()
        if drone_rpy := self.randomization.get("drone_rpy"):
            distrib = getattr(self.np_random, drone_rpy.type)
            kwargs = {k: v for k, v in drone_rpy.items() if k != "type"}
            rpy = np.clip(rpy + distrib(**kwargs), -np.pi, np.pi)

        p.resetBasePositionAndOrientation(
            self.drone.id, pos, p.getQuaternionFromEuler(rpy), physicsClientId=self.pyb_client
        )
        p.resetBaseVelocity(self.drone.id, [0, 0, 0], [0, 0, 0], physicsClientId=self.pyb_client)

    def _sync_pyb_to_sim(self):
        """Read state values from PyBullet and synchronize the drone buffers with it.

        We cache the state values in the drone class to avoid calling PyBullet too frequently.
        """
        pos, quat = p.getBasePositionAndOrientation(self.drone.id, physicsClientId=self.pyb_client)
        self.drone.pos[:] = np.array(pos, float)
        self.drone.rpy[:] = np.array(p.getEulerFromQuaternion(quat), float)
        vel, ang_vel = p.getBaseVelocity(self.drone.id, physicsClientId=self.pyb_client)
        self.drone.vel[:] = np.array(vel, float)
        self.drone.ang_vel[:] = np.array(ang_vel)

    def symbolic(self) -> SymbolicModel:
        """Create a symbolic (CasADi) model for dynamics, observation, and cost.

        Returns:
            CasADi symbolic model of the environment.
        """
        return symbolic(self.drone, 1 / self.settings.sim_freq)

    def _thrust_to_rpm(self, thrust: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert the desired_thrust into motor RPMs.

        Args:
            thrust: The desired thrust per motor.

        Returns:
            The motors' RPMs to apply to the quadrotor.
        """
        thrust = np.clip(thrust, self.drone.params.min_thrust, self.drone.params.max_thrust)
        if "action" in self.disturbances:
            thrust = self.disturbances["action"].apply(thrust)
        thrust = np.clip(thrust, 0, None)  # Make sure thrust is not negative after disturbances
        pwm = (
            np.sqrt(thrust / self.drone.params.kf) - self.drone.params.pwm2rpm_const
        ) / self.drone.params.pwm2rpm_scale
        pwm = np.clip(pwm, self.drone.params.min_pwm, self.drone.params.max_pwm)
        return self.drone.params.pwm2rpm_const + self.drone.params.pwm2rpm_scale * pwm


@dataclass
class SimSettings:
    """Simulation settings dataclass."""

    sim_freq: int = 500
    ctrl_freq: int = 500
    gui: bool = False
    pybullet_id: int = 0
    # Camera settings
    render_resolution: tuple[int, int] = (640, 480)
    camera_view: tuple[float, ...] = (0,) * 16
    camera_projection: tuple[float, ...] = (0,) * 16

    def __post_init__(self):
        """Compute the camera projection and view matrices based on the settings."""
        assert self.sim_freq % self.ctrl_freq == 0, "sim_freq must be divisible by ctrl_freq."
        self.camera_projection = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=self.render_resolution[0] / self.render_resolution[1],
            nearVal=0.1,
            farVal=1000.0,
        )
        self.camera_view = p.computeViewMatrixFromYawPitchRoll(
            distance=3,
            yaw=-30,
            pitch=-30,
            roll=0,
            cameraTargetPosition=[0, 0, 0],
            upAxisIndex=2,
            physicsClientId=0,
        )
