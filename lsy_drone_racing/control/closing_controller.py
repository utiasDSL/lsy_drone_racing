"""This controller is only called when the drone passes the last gate and the environment gets closed.

The controller works in two steps: First, is slows down from any initial speed. Second, it returns to the starting position -> return to home (RTH)

The waypoints for the stopping part are generated using a quintic spline, to be able to set the start and end velocity and acceleration.
The waypoints for the RTH trajectory are created by a cubic spline with waypoints in a predefined height. 
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING, Union

import numpy as np
import pybullet as p
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import BaseController

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ClosingController(BaseController):
    """Controller that creates and follows a braking and homing trajectory."""

    def __init__(self, initial_obs: dict[str, NDArray[np.floating]], initial_info: dict):
        """Initialization of the controller.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        super().__init__(initial_obs, initial_info)

        self._tick = 0
        self.info = initial_info
        self._freq = self.info["env_freq"]
        self._t_step_ctrl = 1/self._freq # control step

        self.debug = False # print statements + makes the tracks plot in sim
        self._obs_x = []
        self._obs_v = []
        self._cmd_x = []
        self._cmd_v = []
        self._cmd_a = []

        self._x_end = 2.0 # distance the drone is supposed to stop behind the last gate
        self._z_homing = 2.0 # height of the return path

        self._mode = 0 # 0=stopping, 1=hover, 2=RTH => order will be 0-1-2-1
        self._target_pos = initial_obs["pos"]
        self._trajectory, self._t_brake = self._generate_stop_trajectory(initial_obs)

        self._t_hover = 2.0 # time the drone hovers before RTH and before landing
        self._t_RTH = 9.0 # time it takes to get back to start (return to home)
        
        self.t_total = self._t_brake + self._t_hover + self._t_RTH + self._t_hover        
        

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        # t = min(self._tick * self._t_step_ctrl, self._t_total)
        t = self._tick * self._t_step_ctrl

        # check if we need to switch modes
        if t <= self._t_brake and self._mode != 0:
            self._mode = 0
        elif t > self._t_brake and t <= self._t_brake+self._t_hover and self._mode != 1:
            self._mode = 1
        elif t > self._t_brake+self._t_hover and t <= self._t_brake+self._t_hover+self._t_RTH and self._mode != 2:
            self._trajectory = self._generate_RTH_trajectory(obs, t)
            self._mode = 2
        elif t > self._t_brake+self._t_hover+self._t_RTH and self._mode != 1:
            self._mode = 1

        # Sample the correct trajectory
        if self._mode == 0: # stopping
            target_pos = self._trajectory(t, order=0)
            target_vel = self._trajectory(t, order=1)
            target_acc = self._trajectory(t, order=2)
            self._target_pos = obs["pos"] # store for a switch to mode 1
        elif self._mode == 2: # RTH
            target_pos = self._trajectory(t)
            trajectory_v = self._trajectory.derivative()
            trajectory_a = trajectory_v.derivative()
            target_vel = trajectory_v(t)*0
            target_acc = trajectory_a(t)*0
            self._target_pos = obs["pos"] # store for a switch to mode 1
        else: # hover
            target_pos = self._target_pos
            target_vel = [0,0,0]
            target_acc = [0,0,0]

        if self.debug:
            self._obs_x.append(obs["pos"])
            self._obs_v.append(obs["vel"])
            self._cmd_x.append(target_pos)
            self._cmd_v.append(target_vel)
            self._cmd_a.append(target_acc)

        # return np.concatenate((target_pos, np.zeros(10)))
        return np.concatenate((target_pos, target_vel, target_acc, np.zeros(4)))

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        """Increment the time step counter."""
        self._tick += 1

    def episode_reset(self):
        """Reset the time step counter."""
        self._tick = 0

    def _generate_stop_trajectory(self, obs: dict[str, NDArray[np.floating]])->Union[QuinticSpline,np.floating]:
        start_pos = obs["pos"]
        start_vel = obs["vel"]
        start_acc = obs["acc"] # TODO

        direction = start_vel/np.linalg.norm(start_vel) # unit vector in the direction of travel
        direction_angle = np.arccos( (np.dot(direction, [0,0,1])) / (1*1) ) 
        direction_angle = -(direction_angle-np.pi/2) # angle to the floor => negative means v_z < 0
        
        # drone can actually go further to no reach the x_end limit if angle!=0
        self._x_end = self._x_end/np.cos(direction_angle)
        # check if drone would crash into floor or ceiling
        x_end_z = start_pos[2] + self._x_end*np.sin(direction_angle)
        if x_end_z < 0.2: 
            if self.debug: 
                print("x_end_z<0.2")
            self._x_end = (0.2 - start_pos[2])/np.sin(direction_angle)
        elif x_end_z > 2.5:
            if self.debug: 
                print("x_end_z>2.5")
            self._x_end = (2.5 - start_pos[2])/np.sin(direction_angle)

        if self.debug: 
            print(f"start_pos_z={start_pos[2]}, x_end_z={start_pos[2] + self._x_end*np.sin(direction_angle)}, x_end={self._x_end}")
            print(f"direction_angle={direction_angle*180/np.pi}°")

        const_acc = np.linalg.norm(start_vel)**2/(self._x_end) # this is just an estimate of what constant deceleration is necessary
        t_brake_max = np.sqrt(4*self._x_end/const_acc) # the time it takes to brake completely
        t_brake = np.arange(0, t_brake_max, self._t_step_ctrl)

        if self.debug:
            print(f"t_brake={t_brake_max}")
            print(f"v_gate = {start_vel}")

        trajectory_stop = QuinticSpline(0, t_brake_max, start_pos, start_vel, start_acc, 
                                        start_pos+direction*self._x_end, start_vel*0, start_acc*0)
        ref_pos_stop = trajectory_stop(t_brake, order=0)

        if self.debug:
            try:
                step = 5
                for i in np.arange(0, len(ref_pos_stop[:,0]) - step, step):
                    p.addUserDebugLine(
                        ref_pos_stop[i,:],
                        ref_pos_stop[i + step,:],
                        lineColorRGB=[0, 1, 0],
                        lineWidth=2,
                        lifeTime=0,  # 0 means the line persists indefinitely
                        physicsClientId=0,
                    )
            except p.error:
                print("PyBullet not available") # Ignore errors if PyBullet is not available
        
        return trajectory_stop, t_brake_max
    
    def _generate_RTH_trajectory(self, obs: dict[str, NDArray[np.floating]], t: np.floating)->CubicSpline:
        start_pos = obs["pos"]

        landing_pos = np.array(self.info["drone_start_pos"]) + np.array([0, 0, 0.25]) # 0.25m above actual landing pos

        intermed_delta = landing_pos - start_pos
        intermed_pos1 = [start_pos[0] + intermed_delta[0]/4, start_pos[1] + intermed_delta[1]/4, self._z_homing]
        intermed_pos2 = [intermed_pos1[0] + intermed_delta[0]/2, intermed_pos1[1] + intermed_delta[1]/2, intermed_pos1[2]]
        intermed_pos3 = [0,0,0]
        intermed_pos3[0] = (5*landing_pos[0]+intermed_pos2[0])/6
        intermed_pos3[1] = (5*landing_pos[1]+intermed_pos2[1])/6
        intermed_pos3[2] = (landing_pos[2]+intermed_pos2[2])/2

        waypoints = np.array([
                start_pos,
                intermed_pos1,
                intermed_pos2,
                intermed_pos3,
                landing_pos,
            ])
        trajectory_RTH = CubicSpline(np.linspace(t, t + self._t_RTH, len(waypoints)), waypoints, bc_type=((1, [0,0,0]), (1, [0,0,0]))) # bc type set boundary conditions for the derivative (here 1)
        
        if self.debug:
            t_plot = np.arange(t, t + self._t_RTH, self._t_step_ctrl) # just for plotting
            ref_pos_return = trajectory_RTH(t_plot)
            try:
                step = 5
                for i in np.arange(0, len(ref_pos_return) - step, step):
                    p.addUserDebugLine(
                        ref_pos_return[i],
                        ref_pos_return[i + step],
                        lineColorRGB=[0, 0, 1],
                        lineWidth=2,
                        lifeTime=0,  # 0 means the line persists indefinitely
                        physicsClientId=0,
                    )
                p.addUserDebugText("x", start_pos, textColorRGB=[0,1,0])
                p.addUserDebugText("x", intermed_pos1, textColorRGB=[0,0,1])
                p.addUserDebugText("x", intermed_pos2, textColorRGB=[0,0,1])
                p.addUserDebugText("x", intermed_pos3, textColorRGB=[0,0,1])
                p.addUserDebugText("x", landing_pos, textColorRGB=[0,0,1])
            except p.error:
                print("PyBullet not available") # Ignore errors if PyBullet is not available
        
        return trajectory_RTH



class QuinticSpline:
    """This class and the code below is mostly written by ChatGPT 4.0."""
    def __init__(self, t0: np.floating, t1: np.floating, x0: np.floating, v0: np.floating, a0: np.floating, x1: np.floating, v1: np.floating, a1: np.floating):
        """Initialize the quintic spline for multidimensional conditions.
        
        Parameters:
        - t0, t1: Start and end times
        - x0, v0, a0: Lists of initial positions, velocities, and accelerations
        - x1, v1, a1: Lists of final positions, velocities, and accelerations
        """
        self.t_points = (t0, t1)  # Start and end time points
        self.dimensions = len(x0)  # Number of dimensions
        self.boundary_conditions = np.array([x0, v0, a0, x1, v1, a1])  # Boundary conditions per dimension
        self.splines = [self._compute_spline(i) for i in range(self.dimensions)]

    def _compute_spline(self, dim: int) -> NDArray:
        t0, t1 = self.t_points
        x0, v0, a0, x1, v1, a1 = self.boundary_conditions[:, dim]

        # Constructing the coefficient matrix for the quintic polynomial
        M = np.array([
            [1, t0, t0**2, t0**3, t0**4, t0**5],            # position @ t0
            [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],        # velocity @ t0
            [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],           # acceleration @ t0
            [1, t1, t1**2, t1**3, t1**4, t1**5],            # position @ t1
            [0, 1, 2*t1, 3*t1**2, 4*t1**3, 5*t1**4],        # velocity @ t1
            [0, 0, 2, 6*t1, 12*t1**2, 20*t1**3],           # acceleration @ t1
        ])

        # Construct the boundary condition vector
        b = np.array([x0, v0, a0, x1, v1, a1])

        # Solve for coefficients of the quintic polynomial
        coefficients = np.linalg.solve(M, b)
        return coefficients

    def __call__(self, t: np.floating, order: int=0) -> NDArray:
        """Evaluate the quintic spline or its derivatives at a given time t for all dimensions.
        
        Parameters:
        - t: Time at which to evaluate
        - order: Derivative order (0=position, 1=velocity, 2=acceleration)
        
        Returns:
        - A list of evaluated values for each dimension
        """
        results = []
        for coeffs in self.splines:
            if order == 0:  # Position
                results.append(sum(c * t**i for i, c in enumerate(coeffs)))
            elif order == 1:  # Velocity
                results.append(sum(i * c * t**(i-1) for i, c in enumerate(coeffs) if i >= 1))
            elif order == 2:  # Acceleration
                results.append(sum(i * (i-1) * c * t**(i-2) for i, c in enumerate(coeffs) if i >= 2))
            else:
                raise ValueError("Only orders 0 (position), 1 (velocity), and 2 (acceleration) are supported.")
        return np.array(results).T # Transpose to make time the first index


"""
Example code how to use the QuinticSpline class
"""
# # Define the boundary conditions for a 2D case
# t0, t1 = 0, 1
# x0, v0, a0 = [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]  # Initial conditions for x and y
# x1, v1, a1 = [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]  # Final conditions for x and y

# # Create the QuinticSpline object
# spline = QuinticSpline(t0, t1, x0, v0, a0, x1, v1, a1)

# # Evaluate the spline and its derivatives
# t_values = np.linspace(t0, t1, 100)
# pos_values = np.array([spline(t, order=0) for t in t_values])  # Position
# vel_values = np.array([spline(t, order=1) for t in t_values])  # Velocity
# acc_values = np.array([spline(t, order=2) for t in t_values])  # Acceleration

# # Plot the results for each dimension
# fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# axes[0].plot(t_values, pos_values[:, 0], label="Position X")
# axes[0].plot(t_values, pos_values[:, 1], label="Position Y")
# axes[0].set_ylabel("Position")
# axes[0].legend()

# axes[1].plot(t_values, vel_values[:, 0], label="Velocity X")
# axes[1].plot(t_values, vel_values[:, 1], label="Velocity Y")
# axes[1].set_ylabel("Velocity")
# axes[1].legend()

# axes[2].plot(t_values, acc_values[:, 0], label="Acceleration X")
# axes[2].plot(t_values, acc_values[:, 1], label="Acceleration Y")
# axes[2].set_ylabel("Acceleration")
# axes[2].set_xlabel("Time")
# axes[2].legend()

# plt.tight_layout()
# plt.show()
