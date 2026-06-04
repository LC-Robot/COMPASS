import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
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


def load_nodes_based_on_yaml(context, *args, **kwargs):
    """Create planner nodes that depend on launch-time scene YAML values."""
    collision_objects_yaml_path = LaunchConfiguration('collision_objects_yaml_path').perform(context)
    method = LaunchConfiguration('method').perform(context)
    run_id = int(LaunchConfiguration('run_id').perform(context))
    level = int(LaunchConfiguration('level').perform(context))
    scene = int(LaunchConfiguration('scene').perform(context))

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={"ros2_control_hardware_type": LaunchConfiguration("ros2_control_hardware_type").perform(context)},
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .sensors_3d(file_path="config/sensors_kinect_pointcloud.yaml")
        .to_moveit_configs()
    )

    try:
        print(f"[Launch] Loading target position from: {collision_objects_yaml_path}")
        with open(collision_objects_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            position = data['prims']['target_box']['position']
            target_params = {
                'target': {
                    'x': float(position[0]),
                    'y': float(position[1]),
                    'z': float(position[2])
                }
            }
        print(f"[Launch] Loaded target position: x={target_params['target']['x']}, y={target_params['target']['y']}, z={target_params['target']['z']}")
    except (FileNotFoundError, KeyError, IndexError, yaml.YAMLError) as e:
        print(f"[Launch] Failed to load target position from YAML: {e}")
        print("[Launch] Falling back to the default target position.")
        target_params = {'target': {'x': 0.5, 'y': 0.0, 'z': 0.0375}}

    common_planner_params = {
        'world_frame': 'panda_link0', 
        'planning_group': 'panda_arm',
        'run_id': run_id,
        'level': level,
        'scene': scene,
        **target_params
    }
    
    planner_nodes = GroupAction(
        actions=[
            Node(
                package='nbv_explorer', 
                executable='rrt_explorer_node', 
                name='rrt_planner', 
                output='screen',
                condition=IfCondition(PythonExpression([f"'{method}' == 'RRT'"])),
                parameters=[
                    common_planner_params,
                    moveit_config.robot_description, 
                    moveit_config.robot_description_semantic, 
                    moveit_config.robot_description_kinematics
                ]
            ),
            Node(
                package='nbv_explorer', 
                executable='fixed_view_planner_node', 
                name='fixed_view_planner',
                output='screen',
                condition=IfCondition(PythonExpression([f"'{method}' == 'FV'"])),
                parameters=[
                    common_planner_params,
                    moveit_config.robot_description, 
                    moveit_config.robot_description_semantic, 
                    moveit_config.robot_description_kinematics
                ]
            ),
            Node(
                package='nbv_explorer', 
                executable='nbv_explorer_node', 
                name='nbv_planner',
                output='screen',
                condition=IfCondition(PythonExpression([f"'{method}' == 'NBV'"])),
                parameters=[
                    common_planner_params,
                    moveit_config.robot_description, 
                    moveit_config.robot_description_semantic, 
                    moveit_config.robot_description_kinematics
                ]
            ),
            Node(
                package='nbv_explorer', 
                executable='geo_rrt_explorer_node',
                name='geo_rrt_planner',            
                output='screen',
                condition=IfCondition(PythonExpression([f"'{method}' == 'GEO_RRT'"])),
                parameters=[
                    common_planner_params,
                    moveit_config.robot_description, 
                    moveit_config.robot_description_semantic, 
                    moveit_config.robot_description_kinematics
                ]
            ),
        ]
    )
    
    guide_publisher_node = Node(
        package='task_guide_publisher',
        executable='guide_publisher_node',
        name='task_guide_publisher',
        output='screen',
        parameters=[target_params]
    )

    return [planner_nodes, guide_publisher_node]


def generate_launch_description():
    
    compass_root = os.environ.get("COMPASS_ROOT", os.path.expanduser("~/ros_workspace/COMPASS"))
    default_collision_objects_yaml_path = os.path.join(compass_root, "config", "level1", "1.yaml")
    
    declared_arguments = [
        DeclareLaunchArgument('method', default_value='RRT'),
        DeclareLaunchArgument('run_id', default_value='3'),
        DeclareLaunchArgument('level', default_value='1'),
        DeclareLaunchArgument('scene', default_value='1'),
        DeclareLaunchArgument("ros2_control_hardware_type", default_value="isaac"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("use_octomap_builder", default_value="true"),
        DeclareLaunchArgument("use_planners", default_value="true"),
        DeclareLaunchArgument(
            'collision_objects_yaml_path',
            default_value=default_collision_objects_yaml_path,
            description='Path to the YAML file defining collision objects.'
        )
    ]
    
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={"ros2_control_hardware_type": LaunchConfiguration("ros2_control_hardware_type")},
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .sensors_3d(file_path="config/sensors_kinect_pointcloud.yaml")
        .to_moveit_configs()
    )
    
    moveit_node_params = moveit_params_for_launch(moveit_config)

    move_group_node = Node(package="moveit_ros_move_group", executable="move_group", output="screen", parameters=[moveit_node_params])
    rviz_node = Node(
        package="rviz2", executable="rviz2", output="log",
        arguments=["-d", os.path.join(get_package_share_directory("isaac_panda"), "config", "panda_moveit_config.rviz")],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        parameters=[moveit_config.robot_description, moveit_config.robot_description_semantic, moveit_config.robot_description_kinematics, moveit_config.planning_pipelines, moveit_config.joint_limits]
    )
    world2robot_tf_node = Node(package="tf2_ros", executable="static_transform_publisher", name="static_transform_publisher_world_robot", output="log", arguments=["--frame-id", "world", "--child-frame-id", "panda_link0"])
    hand2camera_tf_node = Node(package="tf2_ros", executable="static_transform_publisher", name="static_transform_publisher_hand_camera", output="log", arguments=["0.04", "0.0", "0.04", "0.0", "0.0", "0.0", "panda_hand", "sim_camera"])
    robot_state_publisher = Node(package="robot_state_publisher", executable="robot_state_publisher", name="robot_state_publisher", output="both", parameters=[moveit_config.robot_description])
    ros2_controllers_path = os.path.join(get_package_share_directory("moveit_resources_panda_moveit_config"), "config", "ros2_controllers.yaml")
    ros2_control_node = Node(package="controller_manager", executable="ros2_control_node", parameters=[ros2_controllers_path], remappings=[("/controller_manager/robot_description", "/robot_description")], output="screen")
    joint_state_broadcaster_spawner = Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"])
    panda_arm_controller_spawner = Node(package="controller_manager", executable="spawner", arguments=["panda_arm_controller", "-c", "/controller_manager"])
    panda_hand_controller_spawner = Node(package="controller_manager", executable="spawner", arguments=["panda_hand_controller", "-c", "/controller_manager"])
    octomap_config_file = os.path.join(get_package_share_directory('octomap_builder'), 'config', 'params.yaml')
    octomap_server_node = Node(
        package='octomap_builder',
        executable='octomap_builder_node',
        name='octomap_server',
        output='screen',
        condition=IfCondition(LaunchConfiguration("use_octomap_builder")),
        parameters=[octomap_config_file],
        remappings=[('cloud_in', '/camera/depth/points')]
    )
    motion_planner_server_node = Node(
        package="moveit_planning_service", executable="motion_planner_server", name="motion_planner_server", output="screen",
        parameters=[moveit_node_params, {'collision_objects_yaml_path': LaunchConfiguration('collision_objects_yaml_path')}]
    )
    exploration_coordinator_node = Node(
        package='exploration_decision', executable='exploration_coordinator', name='exploration_coordinator', output='screen',
        parameters=[{'method_name': LaunchConfiguration('method'), 'run_id': LaunchConfiguration('run_id'), 'level': LaunchConfiguration('level'), 'scene': LaunchConfiguration('scene')}]
    )

    load_dynamic_nodes_action = OpaqueFunction(
        function=load_nodes_based_on_yaml,
        condition=IfCondition(LaunchConfiguration("use_planners")),
    )

    nodes_to_start = [
        move_group_node,
        rviz_node,
        robot_state_publisher,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        panda_arm_controller_spawner,
        panda_hand_controller_spawner,
        world2robot_tf_node,
        hand2camera_tf_node,
        motion_planner_server_node,
        octomap_server_node,
        exploration_coordinator_node,
        load_dynamic_nodes_action,
    ]
    
    return LaunchDescription(declared_arguments + nodes_to_start)
