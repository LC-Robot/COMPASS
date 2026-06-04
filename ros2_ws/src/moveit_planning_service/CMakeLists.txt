cmake_minimum_required(VERSION 3.8)
project(moveit_planning_service)

# 1. 查找所有必需的包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometric_shapes REQUIRED)
# <--- 修改点 1: 使用正确的 ROS 2 vendor 包名 --->
find_package(yaml_cpp_vendor REQUIRED)

# 2. 从 .action 文件生成 C++ 头文件
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveToPose.action"
  "action/MoveToJoints.action"
  DEPENDENCIES geometry_msgs sensor_msgs
)

# 3. 定义 C++ 可执行文件
add_executable(motion_planner_server src/move_to_pose_action_server.cpp)

# 4. 链接【外部】ROS 包依赖项
ament_target_dependencies(motion_planner_server
  PUBLIC
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  tf2_eigen
  geometric_shapes
)

# ==================【主要修改点】==================
# 将所有剩余的链接库合并到一个使用关键字签名的 target_link_libraries 命令中
rosidl_get_typesupport_target(rosidl_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(motion_planner_server
  PRIVATE
  ${rosidl_typesupport_target}
  Eigen3::Eigen
  # <--- 修改点 2: 使用 vendor 包提供的变量进行链接 --->
  ${yaml_cpp_vendor_LIBRARIES}
)
# ===============================================

# 7. 定义安装规则
install(TARGETS
  motion_planner_server
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/test_move_to_pose_client.py
  scripts/test_joint_move_client.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY action
  DESTINATION share/${PROJECT_NAME}/
)

# 8. 终结包的配置
ament_package()