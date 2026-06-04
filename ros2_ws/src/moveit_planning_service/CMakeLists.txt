cmake_minimum_required(VERSION 3.8)
project(moveit_planning_service)

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
find_package(yaml_cpp_vendor REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveToPose.action"
  "action/MoveToJoints.action"
  DEPENDENCIES geometry_msgs sensor_msgs
)

add_executable(motion_planner_server src/move_to_pose_action_server.cpp)

ament_target_dependencies(motion_planner_server
  PUBLIC
  rclcpp
  rclcpp_action
  moveit_ros_planning_interface
  tf2_eigen
  geometric_shapes
)

rosidl_get_typesupport_target(rosidl_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(motion_planner_server
  PRIVATE
  ${rosidl_typesupport_target}
  Eigen3::Eigen
  ${yaml_cpp_vendor_LIBRARIES}
)

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

ament_package()
