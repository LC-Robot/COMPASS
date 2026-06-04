import os
import yaml # <--- 新增 1: 导入 PyYAML 库
from launch import LaunchDescription
# <--- 新增 2: 导入 OpaqueFunction --- >
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


# <--- 新增 3: 定义一个函数来加载、解析并创建依赖于YAML的节点 --->
def load_nodes_based_on_yaml(context, *args, **kwargs):
    """
    这个函数会在LaunchConfiguration被解析后执行。
    它负责：
    1. 读取 collision_objects_yaml_path 指定的YAML文件。
    2. 从中提取 target_box 的位置。
    3. 创建并返回所有依赖于该位置的节点列表。
    """
    # 首先，获取所有需要的LaunchConfiguration的真实值
    collision_objects_yaml_path = LaunchConfiguration('collision_objects_yaml_path').perform(context)
    method = LaunchConfiguration('method').perform(context)
    run_id = int(LaunchConfiguration('run_id').perform(context))
    level = int(LaunchConfiguration('level').perform(context))
    scene = int(LaunchConfiguration('scene').perform(context))

    # MoveIt的配置也需要在这里重新获取，因为它在函数外部
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

    # 从YAML文件中读取 target_params
    try:
        print(f"[Launch] callback: 正在从以下路径加载目标位置: {collision_objects_yaml_path}")
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
        print(f"[Launch] callback: 成功加载目标位置: x={target_params['target']['x']}, y={target_params['target']['y']}, z={target_params['target']['z']}")
    except (FileNotFoundError, KeyError, IndexError, yaml.YAMLError) as e:
        print(f"[Launch] callback: 错误: 无法从YAML加载目标位置: {e}")
        print("[Launch] callback: 将使用默认的目标位置。")
        # 提供一个备用的默认值，以防文件读取失败
        target_params = {'target': {'x': 0.5, 'y': 0.0, 'z': 0.0375}}

    # =========================================================================
    # == 现在，创建所有依赖 target_params 的节点
    # =========================================================================

    # 创建一个通用的参数字典，可以被多个规划器节点共享
    common_planner_params = {
        'world_frame': 'panda_link0', 
        'planning_group': 'panda_arm',
        'run_id': run_id,
        'level': level,
        'scene': scene,
        **target_params # 将动态加载的 target_params 合并进来
    }
    
    # 规划器节点组 (与之前逻辑相同)
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
    
    # 引导发布节点
    guide_publisher_node = Node(
        package='task_guide_publisher',
        executable='guide_publisher_node',
        name='task_guide_publisher',
        output='screen',
        parameters=[target_params] # 这里也使用动态加载的参数
    )

    # OpaqueFunction 必须返回一个节点/动作的列表
    return [planner_nodes, guide_publisher_node]


def generate_launch_description():
    
    compass_root = os.environ.get("COMPASS_ROOT", os.path.expanduser("~/ros_workspace/COMPASS"))
    default_collision_objects_yaml_path = os.path.join(compass_root, "config", "level1", "1.yaml")
    
    # =================================================================================
    # == 1. 声明所有启动参数 (这部分保持不变)
    # =================================================================================
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
    
    # =================================================================================
    # == 2. 加载MoveIt配置 (只加载一次，然后传递给OpaqueFunction)
    # =================================================================================
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
    
    # =================================================================================
    # == 3. 定义所有不依赖于动态参数的节点 (这部分基本保持不变)
    # =================================================================================
    
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

    # <--- 修改 4: 创建 OpaqueFunction 动作 --->
    # 这个动作会调用我们的函数来生成依赖于YAML的节点
    load_dynamic_nodes_action = OpaqueFunction(
        function=load_nodes_based_on_yaml,
        condition=IfCondition(LaunchConfiguration("use_planners")),
    )

    # =================================================================================
    # == 5. 组合并返回LaunchDescription
    # =================================================================================
    nodes_to_start = [
        # 所有不依赖YAML的静态节点
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
        
        # <--- 修改 5: 添加 OpaqueFunction 动作 --->
        # 它会负责启动 planner_nodes 和 guide_publisher_node
        load_dynamic_nodes_action,
    ]
    
    return LaunchDescription(declared_arguments + nodes_to_start)
