import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

# Helper function to load yaml files
def load_yaml(package_name, file_path):
    """
    Helper function to load a YAML file from a ROS 2 package.
    """
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except (EnvironmentError, yaml.YAMLError) as e:
        print(f"Error loading YAML file '{absolute_file_path}': {e}")
        return None

def generate_launch_description():

    # 1. 声明一个可以在命令行中设置的启动参数
    #    我们将这个“声明对象”存储在名为 ros2_control_hardware_type_arg 的变量中
    ros2_control_hardware_type_arg = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="isaac",
        description="ROS2 control hardware interface type [mock_components, isaac]",
    )

    # 2. 复用你的MoveIt配置
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            # 3. 将启动参数的值传递给 xacro
            mappings={
                "ros2_control_hardware_type": LaunchConfiguration(
                    "ros2_control_hardware_type"
                )
            },
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .to_moveit_configs()
    )

    # --- 加载rrt_explorer节点的特定参数 ---
    rrt_explorer_params = load_yaml("nbv_explorer", "config/rrt_explorer_params.yaml")


    # --- 创建并配置rrt_explorer_node ---
    rrt_explorer_node = Node(
        package="nbv_explorer",
        executable="rrt_explorer_node",
        name="rrt_explorer_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            rrt_explorer_params
        ],
    )

    # --- 4. 返回包含所有组件的LaunchDescription ---
    #    确保这里使用的变量名与步骤1中定义的变量名完全一致
    return LaunchDescription([
        ros2_control_hardware_type_arg,  # <--- 已修正！现在与步骤1的变量名匹配
        rrt_explorer_node
    ])