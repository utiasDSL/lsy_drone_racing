"""
Everything below this comment is entirely written by ChatGPT 4.0
"""

import numpy as np
import matplotlib.pyplot as plt

class QuinticSpline:
    def __init__(self, t0, t1, x0, v0, a0, x1, v1, a1):
        """
        Initialize the quintic spline for multidimensional conditions.
        
        Parameters:
        - t0, t1: Start and end times
        - x0, v0, a0: Lists of initial positions, velocities, and accelerations
        - x1, v1, a1: Lists of final positions, velocities, and accelerations
        """
        self.t_points = (t0, t1)  # Start and end time points
        self.dimensions = len(x0)  # Number of dimensions
        self.boundary_conditions = np.array([x0, v0, a0, x1, v1, a1])  # Boundary conditions per dimension
        self.splines = [self._compute_spline(i) for i in range(self.dimensions)]

    def _compute_spline(self, dim):
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

    def __call__(self, t, order=0):
        """
        Evaluate the quintic spline or its derivatives at a given time t for all dimensions.
        
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
        return np.array(results).T

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
