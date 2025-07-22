# Autonomous Drone Racing with MPC Control

A high-performance Model Predictive Control (MPC) system for autonomous drone racing through dynamic gate sequences. This system implements sophisticated trajectory planning and control that can adapt to moving gates in real-time while optimizing flight performance.

## 🚁 Core Functionality

The system combines several advanced control techniques:

- **Model Predictive Control (MPC)**: Uses ACADOS for fast optimization with attitude control interface
- **Dynamic Trajectory Planning**: Real-time replanning when gates move or are updated
- **Collision Avoidance**: Ellipsoidal constraints for gates and cylindrical constraints for obstacles
- **Adaptive Flight Modes**: Different speed profiles and approach strategies for each gate
- **Momentum Preservation**: Smooth trajectory transitions that avoid abrupt direction changes

## 🛠️ Prerequisites
### Setup

Follow the getting started guide: https://lsy-drone-racing.readthedocs.io/en/latest/getting_started/setup.html

## Acados
Summary of installation:
```bash
# Clone the repo and check out the correct branch, initialize submodules.
cd ~/repos
git clone https://github.com/acados/acados.git
cd acados
git checkout tags/v0.5.0
git submodule update --recursive --init

# Build the application
# Note: If you use Robostack, this might lead to issues. Try to build acados outside your environment if this is the case.
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
# add more optional arguments e.g. -DACADOS_WITH_DAQP=ON, a list of CMake options is provided below
make install -j4

# In your environment, make sure you install the acados python interface:
# Note: If you build acados outside your environment previously, activate it again before executing the following commands.
cd ~/repos/acados
pip install -e interfaces/acados_template

# Make sure acados can be found by adding its location to the path. For robostack and micromamba, this would be:
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$HOME/repos/acados/lib"' > ~/micromamba/envs/ros_env/etc/conda/activate.d/xcustom_acados_ld_library.sh
echo 'export ACADOS_SOURCE_DIR="$HOME/repos/acados"' > ~/micromamba/envs/ros_env/etc/conda/activate.d/xcustom_acados_source.sh


# Deactivate and activate your env again such that the previous two lines can take effect.
micromamba deactivate 
micromamba activate ros_env
# Run a simple example from the acados example to make sure it works.
# If he asks you whether you want to get the t_renderer package installed automatically, press yes.
python3 ~/repos/acados/examples/acados_python/getting_started/minimal_example_ocp.py
```

## 🛠️ System Architecture

### Main Components

#### 1. **MPController** (`attitude_mpc_combined.py`)
The main controller class that orchestrates all components:

**Key Functions:**
- `__init__()`: Initializes MPC solver, trajectory planner, collision avoidance, and logging systems
- `compute_control()`: Main control loop that executes MPC optimization with adaptive replanning
- `_check_and_execute_replanning()`: Detects gate movements above threshold (0.05m) and triggers smooth replanning
- `_execute_mpc_control()`: Solves MPC optimization problem and returns attitude commands
- `_update_weights()`: Dynamically adjusts MPC cost weights during replanning phases
- `step_callback()`: Updates controller state and logs performance metrics after each control step

#### 2. **TrajectoryPlanner** (`smooth_trajectory_planner.py`)
Handles waypoint generation and trajectory smoothing:

**Key Functions:**
- `generate_waypoints()`: Creates waypoints through all gates with approach/exit points and height offsets
- `generate_trajectory_from_waypoints()`: Converts waypoints to smooth trajectories using cubic splines with velocity-aware boundary conditions
- `generate_smooth_replanning_waypoints()`: Creates momentum-preserving waypoints for replanning that avoid backward motion
- `loop_path_gen()`: Generates racing line through gates with individual distances and height offsets
- `calculate_adaptive_speeds()`: Computes speed profiles based on gate proximity and flight phase
- `save_trajectories_to_file()`: Saves planned trajectories and actual flight paths for analysis

#### 3. **CollisionAvoidanceHandler** (`collision_avoidance.py`)
Manages collision constraints for gates and obstacles:

**Key Functions:**
- `setup_model()`: Adds collision constraints to ACADOS model using complementary constraints
- `setup_ocp()`: Configures OCP with constraint dimensions and slack variables
- `update_parameters()`: Updates constraint parameters with current gate/obstacle positions in real-time
- `get_gate_ellipsoids()`: Returns ellipsoid parameters (4 per gate) for constraint visualization
- `get_obstacle_cylinders()`: Returns infinite cylinder parameters for obstacle constraints
- `get_active_obstacle_indices()`: Filters obstacles based on ignored indices configuration

#### 4. **FlightLogger** (`logging_setup.py`)
Comprehensive logging system for analysis and debugging:

**Key Functions:**
- `setup_logging()`: Configures file-only logging with timestamped outputs
- `log_initialization()`: Records controller parameters, weights, and configuration at startup
- `log_replanning_event()`: Logs trajectory replanning events with gate position differences
- `log_tracking_performance()`: Records MPC tracking errors and solver performance
- `log_gate_progress()`: Tracks gate passing events and completion status
- `log_episode_summary()`: Generates comprehensive flight statistics and success metrics

## ⚙️ Configuration Parameters

### MPC Parameters
```python
# Core MPC settings
N = 60                    # Prediction horizon steps
T_HORIZON = 2.0          # Prediction horizon time (seconds)
freq = 240               # Control frequency (Hz)
replanning_frequency = 10 # Replanning check frequency (ticks)

# Cost function weights (tunable for performance)
mpc_weights = {
    'Q_pos': 8.0,        # Position tracking weight
    'Q_vel': 0.01,       # Velocity tracking weight  
    'Q_rpy': 0.01,       # Attitude weight
    'Q_thrust': 0.01,    # Thrust weight
    'Q_cmd': 0.01,       # Command tracking weight
    'R': 0.01           # Control regularization weight
}

# Enhanced weights during replanning (temporary boost)
replanning_weights = {
    'Q_pos': 16.0,       # 2x position tracking during replanning
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
approach_dist = [0.2, 0.3, 0.2, 0.1]     # Distance before gate center

# Individual gate exit distances (meters)  
exit_dist = [0.4, 0.15, 0.2, 5.0]        # Distance after gate center

# Height offsets for approach points (meters)
approach_height_offset = [0.01, 0.1, 0.0, 0.0]  # Vertical offset before gates

# Height offsets for exit points (meters)
exit_height_offset = [0.1, 0.0, 0.05, 0.0]      # Vertical offset after gates

# Default values for gates beyond configured indices
default_approach_dist = 0.2              # Default approach distance
default_exit_dist = 0.3                  # Default exit distance
default_approach_height_offset = 0.0     # Default approach height
default_exit_height_offset = 0.0         # Default exit height
```

### Speed Configuration
```python
# Speed profiles for different flight phases (m/s)
base_speed = 1.6         # Normal cruising speed
high_speed = 2.5         # Speed between gates
approach_speed = 1.0     # Speed when approaching gates
exit_speed = 2.0         # Speed when exiting gates
```

### Collision Avoidance Parameters
```python
# Obstacle and gate constraint parameters
obstacle_radius = 0.14          # Obstacle cylinder radius (meters)
gate_length = 0.50             # Gate length (meters)
ellipsoid_radius = 0.12        # Gate ellipsoid radius (meters)
ellipsoid_length = 0.7         # Gate ellipsoid length (meters)

# Replanning threshold
replanning_threshold = 0.05     # Minimum gate movement to trigger replanning (meters)

# Ignored obstacles (list of indices to skip)
ignored_obstacle_indices = []   # Obstacles to ignore for collision avoidance
```

### Trajectory Planning Parameters
```python
# Trajectory generation settings
N_default = 30                  # Default number of trajectory points
T_HORIZON_default = 1.5        # Default horizon time

# Smoothing parameters
momentum_time = 0.3            # Time to preserve momentum during replanning
max_momentum_distance = 1.0    # Maximum momentum preservation distance
velocity_threshold = 0.5       # Minimum velocity for momentum preservation
transition_factor = 0.7        # Blending factor for smooth transitions

# Generation parameters
min_speed_threshold = 0.1      # Minimum allowable speed
min_gate_duration = 1.0        # Minimum time per gate
extra_points_final_gate = 50   # Extra trajectory points after final gate
extra_points_normal = 200      # Extra trajectory points for intermediate gates
```

## 🎯 Advanced Features

### Adaptive Replanning
The system monitors gate positions and automatically replans trajectories when gates move beyond the threshold:

- **Movement Detection**: Compares observed vs. configured gate positions every 10 ticks
- **Approach Detection**: Only replans when drone is approaching the moved gate (prevents unnecessary replanning)
- **Smooth Transitions**: Preserves forward momentum to avoid abrupt direction changes
- **Weight Adjustment**: Temporarily increases path-following weights during replanning phase

### Dynamic Weight Management
The MPC cost function weights adapt based on flight conditions:

```python
# Weight adjustment phases
weight_adjustment_duration = 2.4 * freq  # Duration of enhanced weights (seconds * frequency)

# Gradual transition between weight sets during replanning
def _update_weights(self):
    if self.weights_adjusted:
        ticks_since_start = self._tick - self.weight_adjustment_start_tick
        if ticks_since_start < self.weight_adjustment_duration:
            self.mpc_weights = self.replanning_mpc_weights.copy()  # Enhanced tracking
        else:
            self.mpc_weights = self.original_mpc_weights.copy()    # Return to normal
            self.weights_adjusted = False
```

### Racing Line Optimization
The trajectory planner implements racing line techniques:

- **Gate Center Shifts**: Shifts gate crossing points forward for optimal racing lines
- **Speed Profiles**: Different speeds for approach (1.0 m/s), gate passage (1.0 m/s), and exit phases (2.0 m/s)
- **Momentum Preservation**: Maintains forward momentum through turns using cubic spline velocity boundary conditions

## 📊 Control Pipeline

1. **Initialization**: Load configuration, setup MPC solver, initialize trajectory planner and collision avoidance
2. **Gate Progress Tracking**: Monitor current target gate and gates passed
3. **Replanning Check**: Every 10 ticks, check if gates have moved beyond 0.05m threshold
4. **Weight Update**: Adjust MPC weights based on replanning status
5. **Collision Parameter Update**: Update ellipsoid and cylinder constraints with current positions
6. **MPC Optimization**: Solve for optimal control inputs over prediction horizon
7. **Control Output**: Return attitude commands [thrust, roll, pitch, yaw]
8. **Logging**: Record performance metrics and flight data

## 📈 Performance Optimization

### Key Performance Parameters
- **Prediction Horizon (N=60)**: Balance between performance and computational cost
- **Control Frequency (240 Hz)**: Higher frequency for better tracking, more computation
- **Replanning Frequency (10 ticks)**: Balance between reactivity and stability
- **Cost Weights**: Tune for desired tracking vs. control effort trade-off
