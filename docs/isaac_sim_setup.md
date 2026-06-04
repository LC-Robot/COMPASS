# Isaac Sim Setup

The Isaac entry point is project-owned and lives at:

```text
ros2_ws/src/isaac_panda/launch/isaac_moveit.py
```

Set the Python executable or wrapper used by Isaac Sim:

```bash
export ISAAC_PYTHON=/path/to/isaac/python.sh
```

The launch script accepts ROS parameters:

```bash
$ISAAC_PYTHON ~/ros_workspace/COMPASS/ros2_ws/src/isaac_panda/launch/isaac_moveit.py \
  --ros-args \
  -p scene_yaml_path:=~/ros_workspace/COMPASS/config/level4/2.yaml \
  -p method:=RRT \
  -p level:=4 \
  -p scene:=2 \
  -p run_id:=40
```

The script publishes Isaac joint state topics consumed by `topic_based_ros2_control`:

```text
/isaac_joint_states
/isaac_joint_commands
```
