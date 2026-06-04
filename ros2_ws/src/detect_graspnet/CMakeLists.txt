cmake_minimum_required(VERSION 3.8)
project(detect_graspnet)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找编译接口文件所需的包
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 定义要生成的接口文件
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetGrasp.srv"
)

ament_package()