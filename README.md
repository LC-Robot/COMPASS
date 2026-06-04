# COMPASS: Confined-space Manipulation Planning with Active Sensing Strategy

COMPASS is a ROS 2, MoveIt 2, and Isaac Sim pipeline for active scene exploration and target-driven manipulation with a Franka Panda arm.

The repository contains the project-owned ROS 2 packages, scene configuration files, lightweight Isaac assets, launch scripts, and setup documentation. It does not vendor MoveIt 2 or other third-party ROS workspaces.

![overview](./overview.png)

## Quick Start

Install ROS 2 Humble, build the external MoveIt workspace, then build this project:

```bash
cd ~/ros_workspace/COMPASS/ros2_ws
source /opt/ros/humble/setup.bash
source ~/ros_workspace/moveit_ws/install/setup.bash
colcon build
source install/setup.bash
```

Run the single-scene pipeline:

```bash
export COMPASS_ROOT=~/ros_workspace/COMPASS
export COMPASS_ROS_SETUP=~/ros_workspace/COMPASS/ros2_ws/install/setup.bash
export MOVEIT_SETUP=~/ros_workspace/moveit_ws/install/setup.bash
export ISAAC_PYTHON=/path/to/isaac/python.sh

./scripts/start_exploration_single_test.sh
```

See `docs/installation.md` and `docs/run_pipeline.md` for details.

## License

Project code is released under Apache-2.0 unless noted otherwise. Third-party dependencies keep their own licenses and are not vendored in this repository.
