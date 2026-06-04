# MoveIt Source Build

This repository does not include `ws_moveit`. Build MoveIt in a separate workspace.

```bash
mkdir -p ~/ros_workspace/moveit_ws/src
cd ~/ros_workspace/moveit_ws
vcs import src < ~/ros_workspace/COMPASS/dependencies.repos
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

If a system package version is sufficient for your environment, you can use apt-installed MoveIt packages instead. The source workspace is recommended when you need the same behavior as the original development machine.

If `move_group` fails because the OMPL plugin links to an old `libompl.so`, rebuild the OMPL planner package:

```bash
cd ~/ros_workspace/moveit_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select moveit_planners_ompl --cmake-clean-cache
source install/setup.bash
ldd install/moveit_planners_ompl/lib/libmoveit_ompl_planner_plugin.so | grep ompl
```
