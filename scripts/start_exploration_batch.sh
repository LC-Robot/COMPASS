#!/usr/bin/env bash
set -euo pipefail

COMPASS_ROOT="${COMPASS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMPASS_ROS_SETUP="${COMPASS_ROS_SETUP:-$COMPASS_ROOT/ros2_ws/install/setup.bash}"
MOVEIT_SETUP="${MOVEIT_SETUP:-$HOME/ros_workspace/moveit_ws/install/setup.bash}"
ISAAC_PYTHON="${ISAAC_PYTHON:-python3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-200}"

METHODS=(${METHODS:-NBV GEO_RRT})
LEVELS=(${LEVELS:-1 2 3 4})
SCENES=(${SCENES:-11})
RUNS=(${RUNS:-6 7 8 9 10 11 12 13 14 15 16 17 18 19 20})

cleanup() {
  pkill -SIGINT -f "isaac_moveit.py" || true
  pkill -SIGINT -f "exploration.launch.py" || true
  pkill -SIGINT -f "graspnet_usd.py" || true
  pkill -SIGINT -f "object_detector.py" || true
  pkill -f "ros2 topic pub --once" || true
}

trap cleanup EXIT SIGINT SIGTERM

for method in "${METHODS[@]}"; do
  for level in "${LEVELS[@]}"; do
    for scene in "${SCENES[@]}"; do
      for run in "${RUNS[@]}"; do
        scene_yaml="$COMPASS_ROOT/config/level${level}/${scene}.yaml"
        if [[ ! -f "$scene_yaml" ]]; then
          echo "Skipping missing scene config: $scene_yaml"
          continue
        fi

        echo "Running method=$method level=$level scene=$scene run=$run"
        METHOD="$method" LEVEL="$level" SCENE="$scene" RUN_ID="$run" \
          TIMEOUT_SECONDS="$TIMEOUT_SECONDS" COLLISION_YAML_PATH="$scene_yaml" \
          "$COMPASS_ROOT/scripts/start_exploration_single_test.sh"

        cleanup
        sleep 30
      done
    done
  done
done
