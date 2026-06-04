import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except (EnvironmentError, yaml.YAMLError) as e:
        print(f"Error loading YAML file '{absolute_file_path}': {e}")
        return None


def generate_launch_description():
    ros2_control_hardware_type_arg = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="isaac",
        description="ROS2 control hardware interface type [mock_components, isaac]",
    )

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
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

    rrt_explorer_params = load_yaml("nbv_explorer", "config/rrt_explorer_params.yaml")

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

    return LaunchDescription([
        ros2_control_hardware_type_arg,
        rrt_explorer_node,
    ])
