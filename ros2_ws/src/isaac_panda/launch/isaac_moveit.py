import sys
import os
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray
import threading

import numpy as np
from pathlib import Path

try:
    from isaacsim import SimulationApp
except:
    from omni.isaac.kit import SimulationApp


GRASP_NOW = False
TARGET_GRASP_DATA = None
SHUTDOWN_EVENT = threading.Event()
simulation_app = None
ros_thread = None
ros_context = None


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def get_declared_parameter(node, name, default_value):
    node.declare_parameter(name, default_value)
    value = node.get_parameter(name).value
    if isinstance(default_value, int) and isinstance(value, str):
        return int(value)
    return value


def method_to_folder(method):
    method_map = {
        "RRT": "our_rrt",
        "NBV": "single_nbv",
        "FV": "fixed_view",
        "GEO_RRT": "geo_rrt",
    }
    return method_map.get(method, method.lower())


class GraspSignalSubscriber(Node):
    """Subscribe to the MoveIt grasp completion signal inside Isaac Sim."""
    def __init__(self, context=None):
        super().__init__('isaac_grasp_signal_subscriber', context=context)
        self.success_subscription = self.create_subscription(
            Bool,
            '/grasp_move_success',
            self.success_listener_callback,
            10)
        self.grasp_subscription = self.create_subscription(
            Float64MultiArray,
            '/grasp',
            self.grasp_callback,
            10)
        self.grasp_data_lock = threading.Lock()
        self.latest_grasp_data = None
        self.get_logger().info('Isaac Sim is listening for /grasp_move_success signal.')

    def success_listener_callback(self, msg):
        global GRASP_NOW, TARGET_GRASP_DATA
        if msg.data:
            self.get_logger().info('Received grasp success signal. Preparing for final approach.')
            with self.grasp_data_lock:
                if self.latest_grasp_data is not None:
                    TARGET_GRASP_DATA = list(self.latest_grasp_data)
                    GRASP_NOW = True
                else:
                    self.get_logger().error('Received grasp success signal, but no /grasp data is available.')

    def grasp_callback(self, msg):
        with self.grasp_data_lock:
            self.latest_grasp_data = msg.data
        self.get_logger().info(f"Received and stored a message from /grasp: {list(msg.data)}", once=True)

def spin_ros_node_in_thread(node):
    """Spin the Isaac-local ROS node until shutdown is requested."""
    global SHUTDOWN_EVENT
    executor = rclpy.executors.SingleThreadedExecutor(context=node.context)
    executor.add_node(node)

    while not SHUTDOWN_EVENT.is_set():
        try:
            executor.spin_once(timeout_sec=0.1)
        except rclpy.executors.ExternalShutdownException:
            break

    print("ROS thread is shutting down.")
    if node.context.ok():
        node.destroy_node()


def gripper_action(action_type, arm, world):
    """
    控制夹爪开合，并等待动作完成。
    """

    lfinger_indices = arm.get_dof_index("panda_finger_joint1")
    rfinger_indices = arm.get_dof_index("panda_finger_joint2")
    
    if action_type == "open":
        target_positions = np.array([0.04, 0.04])
    elif action_type == "close":
        target_positions = np.array([0.0, 0.0])
    else:
        print(f"Unknown gripper action: {action_type}")
        return

    arm.set_joint_position_targets(
        positions=target_positions,
        joint_indices=np.array([lfinger_indices, rfinger_indices])
    )

    # # 持续几帧来设置目标，以确保它被物理引擎捕获
    # for _ in range(10):
    #     arm.set_joint_position_targets(
    #         positions=target_positions,
    #         joint_indices=np.array([lfinger_indices, rfinger_indices])
    #     )
    #     world.step(render=True)
    
    # # 等待夹爪到达目标位置
    # start_time = world.current_time
    # timeout = 5.0 # 5秒超时
    # while world.current_time - start_time < timeout:
    #     world.step(render=True)
    #     current_positions = arm.get_joint_positions(joint_indices=np.array([lfinger_indices, rfinger_indices]))
    #     if np.allclose(current_positions, target_positions, atol=0.005):
    #         print(f"Gripper action '{action_type}' completed successfully.")
    #         return
    # print(f"Warning: Gripper action '{action_type}' timed out.")


FRANKA_STAGE_PATH = "/Franka"
COMPASS_ROOT = Path(os.environ.get("COMPASS_ROOT", Path(__file__).resolve().parents[4]))
FRANKA_USD_PATH = os.environ.get("FRANKA_USD_PATH", str(COMPASS_ROOT / "assets" / "franka.usd"))
BANANA_USD_PATH = os.environ.get("BANANA_USD_PATH", str(COMPASS_ROOT / "assets" / "banana.usd"))
DEFAULT_SCENE_YAML_PATH = str(COMPASS_ROOT / "config" / "level1" / "1.yaml")

# GROUND_PLANE_PATH = "/World/groundPlane" # Path managed by world.scene now
GRAPH_PATH = "/ActionGraph"
ISAAC_HEADLESS = env_flag("ISAAC_HEADLESS", False)
ISAAC_RENDER = env_flag("ISAAC_RENDER", not ISAAC_HEADLESS)
ISAAC_MAX_STEPS = int(os.environ.get("ISAAC_MAX_STEPS", "0"))

try:
    CONFIG = {"renderer": "RayTracedLighting", "headless": ISAAC_HEADLESS}
    print(f"Starting Isaac Sim with headless={ISAAC_HEADLESS}, render={ISAAC_RENDER}, max_steps={ISAAC_MAX_STEPS}")
    print(f"Using Franka USD: {FRANKA_USD_PATH}")
    print(f"Using banana USD: {BANANA_USD_PATH}")

    simulation_app = SimulationApp(CONFIG)

    isaac_sim_ge_4_5_version = True

    from isaacsim.core.version import get_version

    is_legacy_isaacsim = len(get_version()[2]) == 4

    from isaacsim.core.api import World
    from isaacsim.core.utils.prims import set_targets  # noqa E402
    from isaacsim.core.utils import (  # noqa E402
        extensions,
        prims,
        rotations,
        stage,
        viewports,
    )

    from isaacsim.storage.native import nucleus

    from pxr import Gf, UsdGeom  # noqa E402 # Gf is used for rotation
    import omni.graph.core as og  # noqa E402
    import omni

    from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.prims import RigidPrim, GeometryPrim, XFormPrim
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.prims import Articulation
    from isaacsim.core.api.robots import Robot

    # RMPflow
    from isaacsim.robot_motion.motion_generation.lula import RmpFlow
    from isaacsim.robot_motion.motion_generation.interface_config_loader import (
        load_supported_motion_policy_config,
    )
    from isaacsim.robot_motion.motion_generation.articulation_motion_policy import ArticulationMotionPolicy


    # utils
    from utils.camera_utils import setup_camera, setup_camera_publishers, cleanup_camera_pose_publisher

    # Enable Isaac's ROS2 bridge before creating the script-local rclpy context.
    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    ros_context = rclpy.context.Context()
    rclpy.init(args=sys.argv, context=ros_context)

    ros_subscriber_node = GraspSignalSubscriber(context=ros_context)

    method = get_declared_parameter(ros_subscriber_node, "method", "RRT")
    level = get_declared_parameter(ros_subscriber_node, "level", 1)
    scene = get_declared_parameter(ros_subscriber_node, "scene", 1)
    run_id = get_declared_parameter(ros_subscriber_node, "run_id", 1)
    default_scene_yaml_path = str(COMPASS_ROOT / "config" / f"level{level}" / f"{scene}.yaml")
    scene_yaml_path = get_declared_parameter(
        ros_subscriber_node,
        "scene_yaml_path",
        default_scene_yaml_path if os.path.exists(default_scene_yaml_path) else DEFAULT_SCENE_YAML_PATH,
    )

    experiments_root = Path(os.environ.get("COMPASS_EXPERIMENTS_DIR", COMPASS_ROOT / "experiments"))
    result_path = experiments_root / method_to_folder(method) / f"level{level}" / f"scene_{scene}" / f"run_{run_id}"
    result_path.mkdir(parents=True, exist_ok=True)

    ros_subscriber_node.get_logger().info("=================================================")
    ros_subscriber_node.get_logger().info("Isaac Sim parameters:")
    ros_subscriber_node.get_logger().info(f"  - scene_yaml_path: {scene_yaml_path}")
    ros_subscriber_node.get_logger().info(f"  - method: {method}")
    ros_subscriber_node.get_logger().info(f"  - level: {level}")
    ros_subscriber_node.get_logger().info(f"  - scene: {scene}")
    ros_subscriber_node.get_logger().info(f"  - run_id: {run_id}")
    ros_subscriber_node.get_logger().info(f"  - result_path: {result_path}")

    try:
        with open(scene_yaml_path, "r", encoding="utf-8") as f:
            scene_config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        ros_subscriber_node.get_logger().error(f"Scene YAML file not found: {scene_yaml_path}")
        raise
    except yaml.YAMLError as exc:
        ros_subscriber_node.get_logger().error(f"Failed to parse scene YAML {scene_yaml_path}: {exc}")
        raise

    prims_data = scene_config.get("prims", {})

    ros_thread = threading.Thread(target=spin_ros_node_in_thread, args=(ros_subscriber_node,), daemon=True)
    ros_thread.start()

    # --- Using World Class ---
    world = World(stage_units_in_meters=1.0) 

    viewports.set_camera_view(eye=np.array([1.2, 1.2, 0.8]), target=np.array([0, 0, 0.5]))

    world.scene.add_default_ground_plane()

    add_reference_to_stage(usd_path=FRANKA_USD_PATH, prim_path=FRANKA_STAGE_PATH)
    # 获取对机械臂关节的直接API访问权限
    arm = Articulation(prim_paths_expr="/Franka", name="Franka")
    my_franka = world.scene.add(Robot(prim_path="/Franka", name="Franka"))

    simulation_app.update()

    try:
        ros_domain_id = int(os.environ["ROS_DOMAIN_ID"])
        print("Using ROS_DOMAIN_ID: ", ros_domain_id)
    except ValueError:
        print("Invalid ROS_DOMAIN_ID integer value. Setting value to 0")
        ros_domain_id = 0
    except KeyError:
        print("ROS_DOMAIN_ID environment variable is not set. Setting value to 0")
        ros_domain_id = 0

    # --- Action Graph (Remains the same logic) ---
    og_keys_set_values = [
        ("Context.inputs:domain_id", ros_domain_id),
        ("ArticulationController.inputs:robotPath", FRANKA_STAGE_PATH),
        ("PublishJointState.inputs:topicName", "isaac_joint_states"),
        ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
    ]

    if is_legacy_isaacsim:
        og_keys_set_values.insert(
            1, ("ArticulationController.inputs:usePath", True)
        )

    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: og_keys_set_values,
        },
    )

    # --- Set targets for remaining nodes ---
    set_targets(
            prim=stage.get_current_stage().GetPrimAtPath("/ActionGraph/PublishJointState"),
            attribute="inputs:targetPrim",
            target_prim_paths=[FRANKA_STAGE_PATH],
        )

    # 添加灯光
    light_prim = create_prim("/DomeLight", "DomeLight")
    light_prim.GetAttribute("inputs:intensity").Set(1000)

    obstacle_walls = []
    for prim_name, prim_data in prims_data.items():
        position = np.array(prim_data.get("position", [0.0, 0.0, 0.0]))
        orientation = np.array(prim_data.get("orientation", [1.0, 0.0, 0.0, 0.0]))
        size = np.array(prim_data.get("size", [1.0, 1.0, 1.0]))

        if prim_name.startswith("my_custom_box") or prim_name.startswith("obstacle_"):
            obstacle = world.scene.add(
                VisualCuboid(
                    prim_path=f"/World/{prim_name}",
                    name=prim_name,
                    position=position,
                    orientation=orientation,
                    scale=size,
                    color=np.array([255, 0, 0]),
                )
            )
            obstacle_walls.append(obstacle)
        elif prim_name.startswith("wall_"):
            world.scene.add(
                VisualCuboid(
                    prim_path=f"/World/{prim_name}",
                    name=prim_name,
                    position=position,
                    orientation=orientation,
                    scale=size,
                )
            )

    target_box_data = prims_data.get("target_box")
    if target_box_data:
        banana_target_position = np.array([target_box_data.get("position", [0.5, 0.5, 0.05])])
        banana_target_orientation = np.array([target_box_data.get("orientation", [1.0, 0.0, 0.0, 0.0])])
    else:
        print("Warning: target_box not found in scene YAML. Using default banana pose.")
        banana_target_position = np.array([[0.5, 0.5, 0.05]])
        banana_target_orientation = np.array([[1.0, 0.0, 0.0, 0.0]])

    add_reference_to_stage(usd_path = BANANA_USD_PATH, prim_path = "/World/Banana1")
    Banana_Rigid = RigidPrim(
        prim_paths_expr = "/World/Banana1",
        name = "Banana",
        positions = banana_target_position,
        scales = np.array([[0.01, 0.01, 0.01]]),
        orientations = banana_target_orientation,
        masses = np.array([0.02])
    )
    GEOMETRY_BANANA1_PATH = "/World/Banana1/_11_banana"
    Banana_Geometry = GeometryPrim(
        prim_paths_expr = GEOMETRY_BANANA1_PATH,
        name = "banana_mesh",
        collisions = [True],
        scales = np.array([[1.0, 1.0, 1.0]]),
    )
    Banana_Geometry.set_collision_approximations(["convexDecomposition"])

    # 设置所有抓取物体的PhysicsMaterial
    Cardbox_PhysicsMaterial = PhysicsMaterial(
        prim_path="/World/Physics_Material", 
        name="physics_material",
        static_friction=2.0, 
        dynamic_friction=2.0
    )
    Banana_Geometry.apply_physics_materials(Cardbox_PhysicsMaterial)

    # 设置RmpFlow
    rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")

    print("RMP Config:", rmp_config)
    
    rmpflow = RmpFlow(**rmp_config)
    
    physics_dt = 1 / 60.0
    articulation_rmpflow = ArticulationMotionPolicy(my_franka, rmpflow, physics_dt)
    
    articulation_controller = my_franka.get_articulation_controller()

    for obstacle in obstacle_walls:
        rmpflow.add_obstacle(obstacle)
        print(f"Added {obstacle.name} to RMPflow obstacles.")

    # Run app update for multiple frames to re-initialize the ROS action graph after setting new prim inputs
    simulation_app.update()
    simulation_app.update()

    # 设置相机
    camera = setup_camera()
    approx_freq = 30
    setup_camera_publishers(camera, approx_freq, context=ros_context)

    render_frame = ISAAC_RENDER

    # Reset the world before starting the main loop
    world.reset()
    simulation_app.update() # Ensure reset state is rendered

    # Keep the simulated gripper open at startup. The upstream Panda hand
    # ros2_control config initializes finger joints at 0.0, which is closed.
    for _ in range(30):
        gripper_action("open", arm, world)
        world.step(render=render_frame)

    # --- Simulation Loop using World ---
    # No need for simulation_context.play() / stop()
    # --- 主循环 ---
    step_count = 0
    while simulation_app.is_running():

        world.step(render=render_frame)
        step_count += 1
        if ISAAC_MAX_STEPS > 0 and step_count >= ISAAC_MAX_STEPS:
            print(f"Reached ISAAC_MAX_STEPS={ISAAC_MAX_STEPS}. Exiting Isaac Sim loop.")
            break

        # --- 检查标志位并切换控制模式 ---
        if GRASP_NOW:
            if TARGET_GRASP_DATA is None:
                print("ERROR: GRASP_NOW was triggered, but TARGET_GRASP_DATA is None.")
                continue

            grasp_success = False
            grasp_lift_count = 0

            # 闭合夹爪
            print("Executing gripper close action via Isaac Sim API...")
            for i in range(50):
                gripper_action("close", arm, world)
                world.step(render=render_frame)

            # 移动到放置位置
            target_position = np.array([banana_target_position[0][0], banana_target_position[0][1], 0.13])
            target_orientation = np.array([0.70711, 0.0, 0.70711, 0.0])

            for i in range(100):
                rmpflow.set_end_effector_target(
                    target_position=target_position, 
                    target_orientation=target_orientation
                )
                rmpflow.update_world()
                actions = articulation_rmpflow.get_next_articulation_action()
                articulation_controller.apply_action(actions)
                banana_positions, _ = Banana_Rigid.get_world_poses()
                if banana_positions[0][2] > 0.06:
                    grasp_lift_count += 1
                world.step(render=render_frame)

            for i in range(50):
                banana_positions, _ = Banana_Rigid.get_world_poses()
                if banana_positions[0][2] > 0.06:
                    grasp_lift_count += 1
                world.step(render=render_frame)

            grasp_success = grasp_lift_count >= 10
            result_file_path = result_path / "result.txt"
            print(f"Saving result to {result_file_path}...")
            try:
                with open(result_file_path, "w", encoding="utf-8") as f:
                    f.write(f"success: {grasp_success}\n")
                    f.write(f"method: {method}\n")
                    f.write(f"level: {level}\n")
                    f.write(f"scene: {scene}\n")
                    f.write(f"run_id: {run_id}\n")
                    f.write(f"scene_yaml_path: {scene_yaml_path}\n")
                print("Result saved successfully.")
            except OSError as exc:
                print(f"ERROR: Could not write result file: {exc}")

            print("Task complete. Initiating shutdown...")
            simulation_app.close()
            
        else:
            # 平时：保持ROS控制器启用
            og.Controller.set(
                og.Controller.attribute(f"{GRAPH_PATH}/OnImpulseEvent.state:enableImpulse"), True
            )

finally:

    # 步骤 1: 通知ROS线程退出
    SHUTDOWN_EVENT.set()
    
    # 步骤 2: 等待ROS线程完全结束
    if ros_thread and ros_thread.is_alive():
        print("Waiting for ROS thread to join...")
        ros_thread.join(timeout=2.0)
        if ros_thread.is_alive():
            print("Warning: ROS thread did not join in time.")

    # 步骤 3: 现在可以安全地关闭 rclpy
    if ros_context is not None and rclpy.ok(context=ros_context):
        print("Shutting down rclpy.")
        rclpy.shutdown(context=ros_context)
    
    if "cleanup_camera_pose_publisher" in globals():
        cleanup_camera_pose_publisher()
    
    # 步骤 4: 关闭 Isaac Sim
    if simulation_app:
        print("Closing Isaac Sim application.")
        simulation_app.close()
    # =============================================================
    # ================== ^^^ 健壮的清理逻辑 ^^^ ====================
    # =============================================================
