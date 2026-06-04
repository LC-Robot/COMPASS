# Camera utilities for Isaac Sim
import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.nodes.scripts.utils import set_target_prims

# ROS2 imports for camera pose publishing
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped
import threading
import numpy as np

# 相机路径定义
EXISTING_CAMERA_PATH = "/Franka/panda_hand/geometry/realsense/realsense/realsense_camera"

class CameraPosePublisher(Node):
    """相机位姿发布器节点"""
    def __init__(self, context=None):
        super().__init__(
            'camera_pose_publisher',
            context=context,
            parameter_overrides=[
                Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True) ]
            )
        self.publisher = self.create_publisher(PoseStamped, '/camera_pose', 10)
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)  # 30Hz
        self.camera = None

    def set_camera(self, camera):
        """设置相机引用"""
        self.camera = camera
        
    def timer_callback(self):
        """定时发布相机位姿"""
        if self.camera is not None:
            try:
                # 获取相机位姿
                position, orientation = self.camera.get_world_pose('ros')
                
                # 创建PoseStamped消息
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "world"
                
                # 设置位置
                pose_msg.pose.position.x = float(position[0])
                pose_msg.pose.position.y = float(position[1])
                pose_msg.pose.position.z = float(position[2])
                
                # 设置方向 (四元数格式: [w, x, y, z] -> [x, y, z, w])
                pose_msg.pose.orientation.x = float(orientation[1])
                pose_msg.pose.orientation.y = float(orientation[2])
                pose_msg.pose.orientation.z = float(orientation[3])
                pose_msg.pose.orientation.w = float(orientation[0])
                
                # 发布消息
                self.publisher.publish(pose_msg)
                
            except Exception as e:
                self.get_logger().error(f'Error publishing camera pose: {e}')


# 全局变量用于存储发布器节点
_camera_pose_publisher = None
_publisher_thread = None


def setup_camera_pose_publisher(camera, context=None):
    """设置相机位姿发布器"""
    global _camera_pose_publisher, _publisher_thread
    
    if context is not None:
        if not context.ok():
            return
    elif not rclpy.ok():
        return
        
    # 创建相机位姿发布器节点
    _camera_pose_publisher = CameraPosePublisher(context=context)
    _camera_pose_publisher.set_camera(camera)
    
    # 在单独线程中运行发布器
    def spin_publisher():
        executor = rclpy.executors.SingleThreadedExecutor(context=_camera_pose_publisher.context)
        executor.add_node(_camera_pose_publisher)
        try:
            while _camera_pose_publisher is not None and _camera_pose_publisher.context.ok():
                executor.spin_once(timeout_sec=0.1)
        except Exception as e:
            print(f"Camera pose publisher error: {e}")
        finally:
            executor.shutdown()
    
    _publisher_thread = threading.Thread(target=spin_publisher, daemon=True)
    _publisher_thread.start()
    
    print("Camera pose publisher started")


def cleanup_camera_pose_publisher():
    """清理相机位姿发布器"""
    global _camera_pose_publisher, _publisher_thread
    
    if _camera_pose_publisher is not None:
        _camera_pose_publisher.destroy_node()
        _camera_pose_publisher = None
    
    if _publisher_thread is not None:
        _publisher_thread = None


###### Camera helper functions for setting up publishers. ########
def publish_camera_info(camera: Camera, freq):
    """发布相机信息"""
    from isaacsim.ros2.bridge import read_camera_info
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_camera_info"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    writer = rep.writers.get("ROS2PublishCameraInfo")
    camera_info = read_camera_info(render_product_path=render_product)
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
        width=camera_info["width"],
        height=camera_info["height"],
        projectionType=camera_info["projectionType"],
        k=camera_info["k"].reshape([1, 9]),
        r=camera_info["r"].reshape([1, 9]),
        p=camera_info["p"].reshape([1, 12]),
        physicalDistortionModel=camera_info["physicalDistortionModel"],
        physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatch" + "IsaacSimulationGate", render_product
    )

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
    return


def publish_pointcloud_from_depth(camera: Camera, freq):
    """从深度图发布点云"""
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_pointcloud" # Set topic name to the camera's name
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    # Note, this pointcloud publisher will convert the Depth image to a pointcloud using the Camera intrinsics.
    # This pointcloud generation method does not support semantic labeled objects.
    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.DistanceToImagePlane.name
    )

    writer = rep.writers.get(rv + "ROS2PublishPointCloud")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_rgb(camera: Camera, freq):
    """发布RGB图像"""
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_rgb"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_depth(camera: Camera, freq):
    """发布深度图像"""
    # The following code will link the camera's render product and publish the data to the specified topic name.
    render_product = camera._render_product_path
    step_size = int(60/freq)
    topic_name = camera.name+"_depth"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                            sd.SensorType.DistanceToImagePlane.name
                        )
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name
    )
    writer.attach([render_product])

    # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_camera_tf(camera: Camera):
    """发布相机TF变换"""
    camera_prim = camera.prim_path

    if not is_prim_path_valid(camera_prim):
        raise ValueError(f"Camera path '{camera_prim}' is invalid.")

    try:
        # Generate the camera_frame_id. OmniActionGraph will use the last part of
        # the full camera prim path as the frame name, so we will extract it here
        # and use it for the pointcloud frame_id.
        camera_frame_id=camera_prim.split("/")[-1]

        # Generate an action graph associated with camera TF publishing.
        ros_camera_graph_path = "/CameraTFActionGraph"

        # If a camera graph is not found, create a new one.
        if not is_prim_path_valid(ros_camera_graph_path):
            (ros_camera_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": ros_camera_graph_path,
                    "evaluator_name": "execution",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
                },
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("IsaacClock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("RosPublisher", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnTick.outputs:tick", "RosPublisher.inputs:execIn"),
                        ("IsaacClock.outputs:simulationTime", "RosPublisher.inputs:timeStamp"),
                    ]
                }
            )

        # Generate 2 nodes associated with each camera: TF from world to ROS camera convention, and world frame.
        og.Controller.edit(
            ros_camera_graph_path,
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("PublishTF_"+camera_frame_id, "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    ("PublishRawTF_"+camera_frame_id+"_world", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishTF_"+camera_frame_id+".inputs:topicName", "/tf"),
                    # Note if topic_name is changed to something else besides "/tf",
                    # it will not be captured by the ROS tf broadcaster.
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:topicName", "/tf"),
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:parentFrameId", camera_frame_id),
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:childFrameId", camera_frame_id+"_world"),
                    # Static transform from ROS camera convention to world (+Z up, +X forward) convention:
                    ("PublishRawTF_"+camera_frame_id+"_world.inputs:rotation", [0.5, -0.5, 0.5, 0.5]),
                ],
                og.Controller.Keys.CONNECT: [
                    (ros_camera_graph_path+"/OnTick.outputs:tick",
                        "PublishTF_"+camera_frame_id+".inputs:execIn"),
                    (ros_camera_graph_path+"/OnTick.outputs:tick",
                        "PublishRawTF_"+camera_frame_id+"_world.inputs:execIn"),
                    (ros_camera_graph_path+"/IsaacClock.outputs:simulationTime",
                        "PublishTF_"+camera_frame_id+".inputs:timeStamp"),
                    (ros_camera_graph_path+"/IsaacClock.outputs:simulationTime",
                        "PublishRawTF_"+camera_frame_id+"_world.inputs:timeStamp"),
                ],
            },
        )
    except Exception as e:
        print(e)

    # Add target prims for the USD pose. All other frames are static.
    set_target_prims(
        primPath=ros_camera_graph_path+"/PublishTF_"+camera_frame_id,
        inputName="inputs:targetPrims",
        targetPrimPaths=[camera_prim],
    )
    return


def setup_camera():
    """设置相机"""

    # Camera on Franka
    camera = Camera(
        prim_path=EXISTING_CAMERA_PATH,
        frequency=30,
        resolution=(1280, 720),
        )
    camera.initialize()

    # # # Fixed Camera
    # camera = Camera(
    #     prim_path = "/World/Fixd_camera",
    #     frequency = 30,
    #     resolution = (1280, 720),
    # )
    # camera.initialize()
    # camera.set_world_pose(position=[1.88014, 0.77824, 0.74567],  orientation=[0.69222, 0.14432, -0.14432, -0.69222], camera_axes='usd')
    # camera.set_focal_length(18.14756)
    # camera.set_focus_distance(400.0)
    # camera.set_horizontal_aperture(20.955)
    # camera.set_vertical_aperture(15.2908)
    # camera.set_clipping_range(near_distance = 0.01, far_distance = 1000000)
    
    return camera


def setup_camera_publishers(camera, freq=30, context=None):
    """设置相机ROS2发布器"""
    publish_camera_tf(camera)
    publish_camera_info(camera, freq)
    publish_rgb(camera, freq)
    publish_depth(camera, freq)
    publish_pointcloud_from_depth(camera, freq)
    setup_camera_pose_publisher(camera, context=context)
