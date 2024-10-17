"""Quadrotor simulation environment for drone racing.

This module provides a high-fidelity simulation setup for quadrotor drones, particularly focused on
drone racing scenarios. It includes:

* A physics-based simulation using PyBullet
* Detailed drone dynamics and control models
* Symbolic representations for dynamics, observations, and cost functions
* Configurable noise and disturbance models

The simulation environment allows for realistic modeling of drone behavior, including aerodynamics,
motor dynamics, and sensor characteristics. It supports both high-fidelity physics simulations and
analytical models for predictable dynamics.

Key components of the simulation include:

* Drone state representation and dynamics
* An exchangeable physics backend with PyBullet as visualizer
* Customizable environmental factors and disturbances
* Symbolic models for advanced control techniques

The physics backend utilizes PyBullet by default for rigid body dynamics simulation, providing
accurate modeling of collisions, constraints, and multi-body interactions. It can also be replaced
with an analytical model of the drone dynamics.

The simulation framework also includes a symbolic model representation using CasADi, enabling
efficient computation of derivatives for optimization-based control methods. This allows for the
implementation of advanced control techniques such as Model Predictive Control (MPC) and trajectory
optimization.

Users can configure various aspects of the simulation, including drone parameters, environmental
conditions, and simulation fidelity.
"""
