# Autonomous Drone Racing with MPC Control

A high-performance Model Predictive Control (MPC) system for autonomous drone racing through dynamic gate sequences. This project implements a sophisticated trajectory planning and control system that can adapt to moving gates in real-time while optimizing flight performance.

## 🚁 Core Functionality

The system combines several advanced control techniques:

- **Model Predictive Control (MPC)**: Uses ACADOS for fast optimization with attitude control interface
- **Dynamic Trajectory Planning**: Real-time replanning when gates move or are updated
- **Collision Avoidance**: Ellipsoidal constraints for gates and cylindrical constraints for obstacles
- **Adaptive Flight Modes**: Different speed profiles and approach strategies for each gate
- **Momentum Preservation**: Smooth trajectory transitions that avoid abrupt direction changes

## 🛠️ Prerequisites

follow insturctions from https://lsy-drone-racing.readthedocs.io/en/latest/getting_started/setup.html

## 📋 System Architecture

### Main Components

#### 1. **MPController** (`attitude_mpc_combined.py`)
The main controller class that orchestrates all components:

```python
class MPController(Controller):
    def __init__(self, obs, info, config):
        # Initialize MPC solver, trajectory planner, collision avoidance
        
    def compute_control(self, obs, info):
        # Main control computation with replanning logic
        
    def step_callback(self, action, obs, reward, terminated, truncated, info):
        # Update controller state after each step
```

**Key Methods:**
- `compute_control()`: Main control loop with adaptive replanning
- `_check_and_execute_replanning()`: Detects gate movements and triggers replanning
- `_execute_mpc_control()`: Solves MPC optimization problem
- `_update_weights()`: Dynamically adjusts MPC cost weights

#### 2. **TrajectoryPlanner** (`smooth_trajectory_planner.py`)
Handles waypoint generation and trajectory smoothing:

```python
class TrajectoryPlanner:
    def generate_waypoints(self, obs, start_gate_idx, elevated_start):
        # Create waypoints through all gates
        
    def generate_trajectory_from_waypoints(self, waypoints, target_gate_idx, use_velocity_aware):
        # Convert waypoints to smooth trajectory using splines
        
    def generate_smooth_replanning_waypoints(self, obs, current_vel, updated_gate_idx, remaining_gates):
        # Create momentum-preserving waypoints for replanning
```

**Features:**
- Cubic spline interpolation for smooth trajectories
- Velocity-aware boundary conditions for replanning
- Adaptive speed profiles for different flight phases
- Racing line optimization with gate center shifts

#### 3. **CollisionAvoidanceHandler** (`collision_avoidance.py`)
Manages collision constraints for gates and obstacles:

```python
class CollisionAvoidanceHandler:
    def setup_model(self, model):
        # Add collision constraints to ACADOS model
        
    def update_parameters(self, ocp_solver, N_horizon, obs):
        # Update constraint parameters with current positions
        
    def get_gate_ellipsoids(self):
        # Return ellipsoid parameters for gate constraints
```

**Constraint Types:**
- **Gate Constraints**: Four ellipsoids per gate (top, bottom, left, right borders)
- **Obstacle Constraints**: Infinite cylinders for static obstacles
- **Dynamic Updates**: Real-time parameter updates as gates move

#### 4. **FlightLogger** (`logging_setup.py`)
Comprehensive logging system for analysis and debugging:

```python
class FlightLogger:
    def log_initialization(self, controller_params, tick):
        # Log controller setup parameters
        
    def log_replanning_event(self, replan_info, tick):
        # Log trajectory replanning events
        
    def log_tracking_performance(self, tracking_info, tick):
        # Log MPC tracking performance
```

## ⚙️ Configuration Parameters

### MPC Parameters
```python
# Core MPC settings
N = 60                    # Prediction horizon steps
T_HORIZON = 2.0          # Prediction horizon time (seconds)
freq = 240               # Control frequency (Hz)

# Cost function weights
mpc_weights = {
    'Q_pos': 8,          # Position tracking weight
    'Q_vel': 0.01,       # Velocity tracking weight  
    'Q_rpy': 0.01,       # Attitude weight
    'Q_thrust': 0.01,    # Thrust weight
    'Q_cmd': 0.01,       # Command tracking weight
    'R': 0.01           # Control regularization weight
}
```

### Gate-Specific Parameters
```python
# Individual gate approach distances (meters)
approach_dist = [0.2, 0.3, 0.2, 0.1]

# Individual gate exit distances (meters)  
exit_dist = [0.4, 0.15, 0.2, 5.0]

# Height offsets for approach points (meters)
approach_height_offset = [0.01, 0.1, 0.0, 0.0]

# Height offsets for exit points (meters)
exit_height_offset = [0.1, 0.0, 0.05, 0.0]
```

### Collision Avoidance Parameters
```python
OBSTACLE_RADIUS = 0.14    # Obstacle cylinder radius (meters)
GATE_LENGTH = 0.50        # Gate length (meters)
ELLIPSOID_RADIUS = 0.12   # Gate ellipsoid radius (meters)
ELLIPSOID_LENGTH = 0.7    # Gate ellipsoid length (meters)
```

## 🎯 Advanced Features

### Adaptive Replanning
The system monitors gate positions and automatically replans trajectories when gates move:

- **Movement Detection**: Compares observed vs. configured gate positions
- **Approach Detection**: Only replans when drone is approaching the moved gate
- **Smooth Transitions**: Preserves momentum to avoid abrupt direction changes
- **Weight Adjustment**: Temporarily increases path-following weights during replanning

### Dynamic Weight Management
The MPC cost function weights adapt based on flight conditions:

```python
# Normal flight weights
original_weights = {...}

# Enhanced path-following during replanning
replanning_weights = {
    'Q_pos': original_weights['Q_pos'] * 2.0  # Increased position tracking
}
```

### Racing Line Optimization
The trajectory planner implements racing line techniques:

- **Gate Center Shifts**: Shifts gate crossing points forward for optimal racing lines
- **Speed Profiles**: Different speeds for approach, gate passage, and exit phases
- **Momentum Preservation**: Maintains forward momentum through turns

## 📊 Logging and Analysis

### Real-time Logging
All flight data is logged to files in `flight_logs/`:

- **Controller Parameters**: Initial setup and configuration
- **Trajectory Updates**: Waypoint generation and replanning events
- **Gate Progress**: Gate passing events and completion status
- **Performance Metrics**: Tracking errors and solver status
- **Episode Summary**: Final flight statistics and success metrics

### Saved Trajectory Data
Planned trajectories and actual flight paths are saved as `.npz` files for post-flight analysis.

## 📈 Performance Optimization

### Key Performance Parameters
- **Prediction Horizon**: Balance between performance and computational cost
- **Control Frequency**: Higher frequency for better tracking, more computation
- **Cost Weights**: Tune for desired tracking vs. control effort trade-off
- **Replanning Frequency**: Balance between reactivity and stability

### Speed Optimization Tips
1. Adjust `approach_dist` and `exit_dist` for faster gate passages
2. Tune speed profiles in `calculate_adaptive_speeds()`
3. Optimize gate center shifts for racing lines
4. Balance momentum preservation vs. trajectory tracking
