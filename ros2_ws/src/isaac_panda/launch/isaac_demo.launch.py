import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def moveit_params_for_launch(moveit_config):
    params = moveit_config.to_dict()
    sensors = params.get("sensors")
    if isinstance(sensors, list) and sensors and isinstance(sensors[0], dict):
        sensor_name = "kinect_pointcloud"
        sensor_params = sensors[0]
        params["sensors"] = [sensor_name]
        for key, value in sensor_params.items():
            params[f"{sensor_name}.{key}"] = value
    for pipeline_name in params.get("planning_pipelines", []):
        pipeline_params = params.get(pipeline_name)
        if isinstance(pipeline_params, dict):
            planning_plugin = pipeline_params.get("planning_plugin")
            if isinstance(planning_plugin, str):
                pipeline_params["planning_plugins"] = [planning_plugin]
            request_adapters = pipeline_params.get("request_adapters")
            if isinstance(request_adapters, str):
                pipeline_params["request_adapters"] = request_adapters.split()
            response_adapters = pipeline_params.get("response_adapters")
            if isinstance(response_adapters, str):
                pipeline_params["response_adapters"] = response_adapters.split()
            request_adapter_map = {
                "default_planner_request_adapters/ResolveConstraintFrames": "default_planning_request_adapters/ResolveConstraintFrames",
                "default_planner_request_adapters/FixWorkspaceBounds": "default_planning_request_adapters/ValidateWorkspaceBounds",
                "default_planner_request_adapters/FixStartStateBounds": "default_planning_request_adapters/CheckStartStateBounds",
                "default_planner_request_adapters/FixStartStateCollision": "default_planning_request_adapters/CheckStartStateCollision",
            }
            migrated_request_adapters = []
            migrated_response_adapters = list(pipeline_params.get("response_adapters", []))
            for adapter in pipeline_params.get("request_adapters", []):
                if adapter == "default_planner_request_adapters/AddTimeOptimalParameterization":
                    migrated_response_adapters.insert(0, "default_planning_response_adapters/AddTimeOptimalParameterization")
                elif adapter in request_adapter_map:
                    migrated_request_adapters.append(request_adapter_map[adapter])
                elif adapter.startswith("default_planning_request_adapters/"):
                    migrated_request_adapters.append(adapter)
            if migrated_request_adapters:
                pipeline_params["request_adapters"] = migrated_request_adapters
            if migrated_response_adapters:
                pipeline_params["response_adapters"] = migrated_response_adapters
    return params


def generate_launch_description():

    # Command-line arguments
    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="isaac",
        description="ROS2 control hardware interface type to use for the launch file -- possible values: [mock_components, isaac]",
    )
    use_rviz = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Whether to launch RViz.",
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
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .sensors_3d(
            file_path=os.path.join(
                get_package_share_directory("moveit_resources_panda_moveit_config"),
                "config/sensors_kinect_pointcloud.yaml",
            )
        )
        .to_moveit_configs()
    )

    path = os.path.join(
                get_package_share_directory("moveit_resources_panda_moveit_config"),
                "config/sensors_kinect_pointcloud.yaml",
            )
    print("path:", path)

    moveit_node_params = moveit_params_for_launch(moveit_config)

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_node_params],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # RViz
    rviz_config_file = os.path.join(
        get_package_share_directory("isaac_panda"),
        "config",
        "panda_moveit_config.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # Static TF
    world2robot_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["--frame-id", "world", "--child-frame-id", "panda_link0"],
    )
    hand2camera_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=[
            "0.04",
            "0.0",
            "0.04",
            "0.0",
            "0.0",
            "0.0",
            "panda_hand",
            "sim_camera",
        ],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # ros2_control using FakeSystem as hardware
    ros2_controllers_path = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            ros2_controllers_path,
        ],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    panda_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_arm_controller", "-c", "/controller_manager"],
    )

    panda_hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_hand_controller", "-c", "/controller_manager"],
    )

    motion_planner_server_node = Node(
        package="moveit_planning_service",
        executable="motion_planner_server",
        name="motion_planner_server",
        output="screen",
        parameters=[moveit_node_params],
    )

    return LaunchDescription(
        [
            ros2_control_hardware_type,
            use_rviz,
            robot_state_publisher,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            panda_arm_controller_spawner,
            panda_hand_controller_spawner,
            world2robot_tf_node,
            hand2camera_tf_node,
            move_group_node,
            rviz_node,
            motion_planner_server_node,
        ]
    )
