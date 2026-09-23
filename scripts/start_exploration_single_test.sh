#!/usr/bin/env bash
set -euo pipefail

COMPASS_ROOT="${COMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
COMPASS_ROS_SETUP="${COMPASS_ROS_SETUP:-$COMPASS_ROOT/ros2_ws/install/setup.bash}"
MOVEIT_SETUP="${MOVEIT_SETUP:-$HOME/ros_workspace/moveit_ws/install/setup.bash}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/le/anaconda3/envs/env_isaaclab/bin/python}"
GRASPNET_ROOT="${GRASPNET_ROOT:-/home/le/ros_workspace/COMPASS/ros2_ws/src/detect_graspnet}"
GRASPNET_CONDA_ENV="${GRASPNET_CONDA_ENV:-graspnet-test}"

LEVEL="${LEVEL:-1}"
SCENE="${SCENE:-1}"
RUN_ID="${RUN_ID:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-200}"
ROS2_CONTROL_HARDWARE_TYPE="${ROS2_CONTROL_HARDWARE_TYPE:-isaac}"
USE_RVIZ="${USE_RVIZ:-true}"
USE_OCTOMAP_BUILDER="${USE_OCTOMAP_BUILDER:-true}"
USE_PLANNERS="${USE_PLANNERS:-true}"
COLLISION_YAML_PATH="${COLLISION_YAML_PATH:-$COMPASS_ROOT/config/level${LEVEL}/${SCENE}.yaml}"
ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/compass_ros_logs}"
COMPASS_EXPERIMENTS_DIR="${COMPASS_EXPERIMENTS_DIR:-$COMPASS_ROOT/experiments}"

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "Missing ROS setup file: $ROS_SETUP" >&2
  exit 1
fi

if [[ ! -f "$COMPASS_ROS_SETUP" ]]; then
  echo "Missing COMPASS setup file: $COMPASS_ROS_SETUP" >&2
  echo "Build the project first: cd $COMPASS_ROOT/ros2_ws && colcon build" >&2
  exit 1
fi

if [[ ! -f "$MOVEIT_SETUP" ]]; then
  echo "Missing MoveIt setup file: $MOVEIT_SETUP" >&2
  echo "Build or install MoveIt first. See the Installation section in README.md." >&2
  exit 1
fi

if [[ ! -f "$COLLISION_YAML_PATH" ]]; then
  echo "Missing scene config: $COLLISION_YAML_PATH" >&2
  exit 1
fi

if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "Missing executable Isaac Python: $ISAAC_PYTHON" >&2
  echo "Override it with: ISAAC_PYTHON=/path/to/isaac/python ./scripts/start_exploration_single_test.sh" >&2
  exit 1
fi

mkdir -p "$ROS_LOG_DIR" "$COMPASS_EXPERIMENTS_DIR"

terminal=(env -u LD_LIBRARY_PATH gnome-terminal)

cleanup() {
  echo "Shutting down COMPASS runtime processes..."
  pkill -SIGINT -f "isaac_moveit.py" || true
  pkill -SIGINT -f "exploration.launch.py" || true
  pkill -SIGINT -f "graspnet_usd.py" || true
  pkill -SIGINT -f "grasp_server_node.py" || true
  pkill -SIGINT -f "object_detector.py" || true
  pkill -f "ros2 topic pub --once /start_exploring" || true
  pkill -f "ros2 topic pub --once /resume_detection" || true
}

trap cleanup EXIT SIGINT SIGTERM

echo "Starting COMPASS single-scene run"
echo "  level=$LEVEL scene=$SCENE run_id=$RUN_ID"
echo "  hardware=$ROS2_CONTROL_HARDWARE_TYPE use_rviz=$USE_RVIZ"
echo "  scene_config=$COLLISION_YAML_PATH"
echo "  compass=$COMPASS_ROOT"
echo "  moveit=$MOVEIT_SETUP"

"${terminal[@]}" --title="[COMPASS] Isaac Sim" -- bash -c "
  export COMPASS_ROOT='$COMPASS_ROOT'
  export ROS_LOG_DIR='$ROS_LOG_DIR'
  export COMPASS_EXPERIMENTS_DIR='$COMPASS_EXPERIMENTS_DIR'
  source '$ROS_SETUP'
  source '$MOVEIT_SETUP'
  source '$COMPASS_ROS_SETUP'
  cd '$COMPASS_ROOT/ros2_ws'
  '$ISAAC_PYTHON' '$COMPASS_ROOT/ros2_ws/src/isaac_panda/launch/isaac_moveit.py' \
    --ros-args \
    -p scene_yaml_path:='$COLLISION_YAML_PATH' \
    -p level:='$LEVEL' \
    -p scene:='$SCENE' \
    -p run_id:='$RUN_ID'
  exec bash
"

sleep 30

"${terminal[@]}" --title="[COMPASS] ROS Launch" -- bash -c "
  export COMPASS_ROOT='$COMPASS_ROOT'
  export ROS_LOG_DIR='$ROS_LOG_DIR'
  export COMPASS_EXPERIMENTS_DIR='$COMPASS_EXPERIMENTS_DIR'
  source '$ROS_SETUP'
  source '$MOVEIT_SETUP'
  source '$COMPASS_ROS_SETUP'
  cd '$COMPASS_ROOT/ros2_ws'
  ros2 launch exploration_decision exploration.launch.py \
    run_id:='$RUN_ID' \
    level:='$LEVEL' \
    scene:='$SCENE' \
    collision_objects_yaml_path:='$COLLISION_YAML_PATH' \
    ros2_control_hardware_type:='$ROS2_CONTROL_HARDWARE_TYPE' \
    use_rviz:='$USE_RVIZ' \
    use_octomap_builder:='$USE_OCTOMAP_BUILDER' \
    use_planners:='$USE_PLANNERS'
  exec bash
"

sleep 5

if [[ -n "$GRASPNET_ROOT" && -f "$GRASPNET_ROOT/detect_graspnet/grasp_server_node.py" ]]; then
  "${terminal[@]}" --title="[COMPASS] GraspNet Node" -- bash -c "
    export COMPASS_ROOT='$COMPASS_ROOT'
    export ROS_LOG_DIR='$ROS_LOG_DIR'
    source '$ROS_SETUP'
    source '$MOVEIT_SETUP'
    source '$COMPASS_ROS_SETUP'
    if command -v conda >/dev/null 2>&1; then
      source \"\$(conda info --base)/etc/profile.d/conda.sh\"
      conda activate '$GRASPNET_CONDA_ENV'
    fi
    cd '$GRASPNET_ROOT/detect_graspnet'
    python3 grasp_server_node.py
    exec bash
  "
else
  echo "Skipping GraspNet Server Node. Set GRASPNET_ROOT to enable it."
fi

sleep 2

"${terminal[@]}" --title="[COMPASS] Object Detector" -- bash -c "
  export COMPASS_ROOT='$COMPASS_ROOT'
  export ROS_LOG_DIR='$ROS_LOG_DIR'
  source '$ROS_SETUP'
  source '$MOVEIT_SETUP'
  source '$COMPASS_ROS_SETUP'
  if command -v conda >/dev/null 2>&1; then
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate '$GRASPNET_CONDA_ENV' || true
  fi
  cd '$COMPASS_ROOT/ros2_ws/src/detect_graspnet/detect_graspnet'
  python3 object_detector.py
  exec bash
"

sleep 5

"${terminal[@]}" --title="[COMPASS] Start Exploration" -- bash -c "
  source '$ROS_SETUP'
  source '$MOVEIT_SETUP'
  source '$COMPASS_ROS_SETUP'
  ros2 topic pub --once /start_exploring std_msgs/msg/Bool '{data: true}'
  read -r -p 'Start signal sent. Press Enter to close.'
"

sleep 3

"${terminal[@]}" --title="[COMPASS] Resume Detection" -- bash -c "
  source '$ROS_SETUP'
  source '$MOVEIT_SETUP'
  source '$COMPASS_ROS_SETUP'
  ros2 topic pub --once /resume_detection std_msgs/msg/Bool '{data: true}'
  read -r -p 'Resume signal sent. Press Enter to close.'
"

echo "All terminals started. Runtime cleanup will run after ${TIMEOUT_SECONDS}s or Ctrl+C."
sleep "$TIMEOUT_SECONDS"
