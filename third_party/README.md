# Third-Party Dependencies

This directory is intentionally documentation-only.

Do not vendor large third-party workspaces such as MoveIt 2, GraspNet Baseline,
GraspNet API, `ros2_kortex`, `ros2_robotiq_gripper`, `serial`, or generated
`build/install/log` directories into this repository.

Use `../dependencies.repos` to fetch the source dependencies into a separate workspace.

GraspNet Baseline is an exception at runtime: `grasp_server_node.py` expects a
local checkout inside the `detect_graspnet` package directory so its legacy
imports resolve. Keep that checkout local and ignored by git. See the
Installation section in `../README.md`.
