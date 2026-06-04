import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("octomap_builder")
    config_file = os.path.join(pkg_share, "config", "params.yaml")

    return LaunchDescription([
        Node(
            package="octomap_builder",
            executable="octomap_builder_node",
            name="octomap_builder_node",
            output="screen",
            parameters=[config_file],
        )
    ])
