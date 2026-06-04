cmake_minimum_required(VERSION 3.8)
project(octomap_builder)

cmake_policy(SET CMP0074 NEW)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找编译所需的依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

# 查找 OctoMap 依赖
find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)

# 需要找到PCL包，并指定我们需要哪些组件(Components)
# IO用于读写，Filters用于体素滤波
find_package(PCL 1.10 REQUIRED COMPONENTS IO Filters)

# 添加可执行文件
add_executable(octomap_builder_node src/octomap_builder_node.cpp)

# 使用 ament_target_dependencies 来处理ROS 2的包
ament_target_dependencies(octomap_builder_node
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_ros
  tf2_sensor_msgs
  pcl_conversions
  octomap_msgs
)

# 使用 target_link_libraries 来链接非ament包（如OctoMap和PCL）
target_link_libraries(octomap_builder_node
  ${octomap_LIBRARIES}
  ${PCL_LIBRARIES} # 将PCL库链接到我们的目标
)

# 安装可执行文件
install(TARGETS
  octomap_builder_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # ament_lint_auto会自动找到大部分测试依赖
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()