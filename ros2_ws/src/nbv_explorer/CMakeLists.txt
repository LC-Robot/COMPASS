cmake_minimum_required(VERSION 3.8)
project(nbv_explorer)

# =========================================================================
# 1. 查找所有必需的包 (保持不变)
# =========================================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(octomap_ros REQUIRED)
find_package(octomap REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

# =========================================================================
# 2. 从 .srv 文件生成接口 (保持不变)
# =========================================================================
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetNBV.srv"
  "srv/GetInitialCoverage.srv"
  "srv/UpdateWeights.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

# =========================================================================
# 3. 【已修改】定义四个可执行文件
# =========================================================================
add_executable(rrt_explorer_node src/rrt_explorer_node.cpp)
add_executable(fixed_view_planner_node src/fixed_view_planner_node.cpp)
add_executable(nbv_explorer_node src/nbv_explorer_node.cpp)
add_executable(geo_rrt_explorer_node src/geo_rrt_explorer_node.cpp)

# =========================================================================
# 4. 【已修改】为每个可执行文件链接AMENT依赖项
# =========================================================================
ament_target_dependencies(rrt_explorer_node
  rclcpp std_msgs geometry_msgs sensor_msgs visualization_msgs octomap_msgs
  moveit_core moveit_ros_planning_interface octomap_ros
  pcl_conversions tf2_ros tf2_eigen
)

ament_target_dependencies(fixed_view_planner_node
  rclcpp std_msgs geometry_msgs sensor_msgs visualization_msgs octomap_msgs
  moveit_core moveit_ros_planning_interface octomap_ros
  pcl_conversions tf2_ros tf2_eigen
)

ament_target_dependencies(nbv_explorer_node
  rclcpp std_msgs geometry_msgs sensor_msgs visualization_msgs octomap_msgs
  moveit_core moveit_ros_planning_interface octomap_ros
  pcl_conversions tf2_ros tf2_eigen
)

# 为新的 geo_rrt 节点添加依赖项
ament_target_dependencies(geo_rrt_explorer_node               # <-- 新增
  rclcpp std_msgs geometry_msgs sensor_msgs visualization_msgs octomap_msgs
  moveit_core moveit_ros_planning_interface octomap_ros
  pcl_conversions tf2_ros tf2_eigen
)

# =========================================================================
# 5. 【已修改】为每个可执行文件链接非AMENT库
# =========================================================================
rosidl_get_typesupport_target(idl_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(rrt_explorer_node
    ${idl_target}
    ${octomap_LIBRARIES}
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

target_link_libraries(fixed_view_planner_node
    ${idl_target}
    ${octomap_LIBRARIES}
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

target_link_libraries(nbv_explorer_node
    ${idl_target}
    ${octomap_LIBRARIES}
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

# 为新的 geo_rrt 节点链接库
target_link_libraries(geo_rrt_explorer_node                     # <-- 新增
    ${idl_target}
    ${octomap_LIBRARIES}
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

# =========================================================================
# 6. 安装规则
# =========================================================================
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY srv
  DESTINATION share/${PROJECT_NAME}/
)

# --- 【已修改】安装四个可执行文件 ---
install(TARGETS
  rrt_explorer_node
  fixed_view_planner_node
  nbv_explorer_node
  geo_rrt_explorer_node # <-- 新增
  DESTINATION lib/${PROJECT_NAME}
)

# =========================================================================
# 7. 终结包的配置
# =========================================================================
ament_package()