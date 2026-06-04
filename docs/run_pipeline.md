# Running The Pipeline

Set the workspace paths:

```bash
export COMPASS_ROOT=~/ros_workspace/COMPASS
export COMPASS_ROS_SETUP=~/ros_workspace/COMPASS/ros2_ws/install/setup.bash
export MOVEIT_SETUP=~/ros_workspace/moveit_ws/install/setup.bash
export ISAAC_PYTHON=/path/to/isaac/python.sh
export GRASPNET_CONDA_ENV=graspnet-test
export COMPASS_YOLO_DEVICE=cuda:0
```

Run a single scene:

```bash
cd ~/ros_workspace/COMPASS
./scripts/start_exploration_single_test.sh
```

Useful optional parameters:

```bash
METHOD=RRT LEVEL=4 SCENE=2 RUN_ID=40 ./scripts/start_exploration_single_test.sh
```

If YOLO-World reports mixed CPU/CUDA tensors in the GraspNet conda
environment, force the detector to CPU for debugging:

```bash
COMPASS_YOLO_DEVICE=cpu ./scripts/start_exploration_single_test.sh
```

The script launches:

1. Isaac Sim through `isaac_panda/launch/isaac_moveit.py`.
2. `ros2 launch exploration_decision exploration.launch.py`.
3. GraspNet node from `detect_graspnet/detect_graspnet/grasp_server_node.py`.
4. Object detector from the `detect_graspnet` package.
5. `/start_exploring` and `/resume_detection` trigger topics.

For headless or non-GNOME systems, launch the commands manually from the script instead of using `gnome-terminal`.
