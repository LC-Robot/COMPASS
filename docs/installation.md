# Installation

This project was organized around ROS 2 Humble, MoveIt 2, and Isaac Sim. The recommended open-source setup uses two workspaces:

```text
~/ros_workspace/moveit_ws     # Third-party MoveIt source workspace
~/ros_workspace/COMPASS       # This repository
```

## System Dependencies

Install ROS 2 Humble and common build tools:

```bash
sudo apt update
sudo apt install python3-colcon-common-extensions python3-vcstool python3-rosdep
sudo rosdep init || true
rosdep update
```

Install MoveIt dependencies from source as described in `docs/moveit_source_build.md`.

## Build COMPASS

```bash
cd ~/ros_workspace/COMPASS/ros2_ws
source /opt/ros/humble/setup.bash
source ~/ros_workspace/moveit_ws/install/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

## Python Environments

Isaac Sim is usually launched with Isaac's Python executable or wrapper. Set:

```bash
export ISAAC_PYTHON=/path/to/isaac/python.sh
```

The detection/grasp components use a separate conda environment. For the
GraspNet node, create a `graspnet-test` environment and install
`graspnet/graspnet-baseline` locally under the `detect_graspnet` package. See
`docs/graspnet_setup.md`.

Model weights are not included in this repository. Place the GraspNet RealSense
checkpoint at:

```text
~/ros_workspace/COMPASS/ros2_ws/src/detect_graspnet/detect_graspnet/logs/log_rs/checkpoint-rs.tar
```
