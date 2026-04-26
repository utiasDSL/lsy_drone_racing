
#!/usr/bin/env bash
mkdir -p ros_ws/src

if [ ! -d ros_ws/src/motion_capture_tracking/.git ]; then
  echo "[Pixi activation] Cloning motion_capture_tracking..."
  git clone --recurse-submodules https://github.com/utiasDSL/motion_capture_tracking ros_ws/src/motion_capture_tracking
fi

if [ ! -d ros_ws/src/drone_racing_msgs/.git ]; then
  echo "[Pixi activation] Cloning drone_racing_msgs..."
  git clone https://github.com/rducrist/drone_racing_msgs.git ros_ws/src/drone_racing_msgs
fi

if [ ! -f ros_ws/install/setup.sh ] || [ ! -d ros_ws/install/drone_racing_msgs ]; then
  echo "[Pixi activation] Running colcon build..."
  (cd ros_ws && colcon build --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
fi

. ./ros_ws/install/setup.sh