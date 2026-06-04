cmake_minimum_required(VERSION 3.8)
project(octomap_builder)

cmake_policy(SET CMP0074 NEW)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)

find_package(PCL 1.10 REQUIRED COMPONENTS IO Filters)

add_executable(octomap_builder_node src/octomap_builder_node.cpp)

ament_target_dependencies(octomap_builder_node
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_ros
  tf2_sensor_msgs
  pcl_conversions
  octomap_msgs
)

target_link_libraries(octomap_builder_node
  ${octomap_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS
  octomap_builder_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
