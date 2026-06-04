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

EXISTING_CAMERA_PATH = "/Franka/panda_hand/geometry/realsense/realsense/realsense_camera"


class CameraPosePublisher(Node):
    """Publish the Isaac camera pose as a ROS PoseStamped message."""

    def __init__(self, context=None):
        super().__init__(
            "camera_pose_publisher",
            context=context,
            parameter_overrides=[
                Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)
            ],
        )
        self.publisher = self.create_publisher(PoseStamped, "/camera_pose", 10)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.camera = None

    def set_camera(self, camera):
        self.camera = camera
        
    def timer_callback(self):
        if self.camera is not None:
            try:
                position, orientation = self.camera.get_world_pose("ros")
                
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "world"
                
                pose_msg.pose.position.x = float(position[0])
                pose_msg.pose.position.y = float(position[1])
                pose_msg.pose.position.z = float(position[2])
                
                pose_msg.pose.orientation.x = float(orientation[1])
                pose_msg.pose.orientation.y = float(orientation[2])
                pose_msg.pose.orientation.z = float(orientation[3])
                pose_msg.pose.orientation.w = float(orientation[0])
                
                self.publisher.publish(pose_msg)
                
            except Exception as e:
                self.get_logger().error(f"Error publishing camera pose: {e}")


_camera_pose_publisher = None
_publisher_thread = None


def setup_camera_pose_publisher(camera, context=None):
    global _camera_pose_publisher, _publisher_thread
    
    if context is not None:
        if not context.ok():
            return
    elif not rclpy.ok():
        return
        
    _camera_pose_publisher = CameraPosePublisher(context=context)
    _camera_pose_publisher.set_camera(camera)
    
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
    global _camera_pose_publisher, _publisher_thread
    
    if _camera_pose_publisher is not None:
        _camera_pose_publisher.destroy_node()
        _camera_pose_publisher = None
    
    if _publisher_thread is not None:
        _publisher_thread = None


def publish_camera_info(camera: Camera, freq):
    from isaacsim.ros2.bridge import read_camera_info

    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_camera_info"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

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

    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)
    return


def publish_pointcloud_from_depth(camera: Camera, freq):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_pointcloud"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.DistanceToImagePlane.name
    )

    writer = rep.writers.get(rv + "ROS2PublishPointCloud")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_rgb(camera: Camera, freq):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_rgb"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_depth(camera: Camera, freq):
    render_product = camera._render_product_path
    step_size = int(60 / freq)
    topic_name = camera.name + "_depth"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.DistanceToImagePlane.name
    )
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    return


def publish_camera_tf(camera: Camera):
    camera_prim = camera.prim_path

    if not is_prim_path_valid(camera_prim):
        raise ValueError(f"Camera path '{camera_prim}' is invalid.")

    try:
        camera_frame_id = camera_prim.split("/")[-1]

        ros_camera_graph_path = "/CameraTFActionGraph"

        if not is_prim_path_valid(ros_camera_graph_path):
            og.Controller.edit(
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
                    ],
                },
            )

        og.Controller.edit(
            ros_camera_graph_path,
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("PublishTF_" + camera_frame_id, "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    ("PublishRawTF_" + camera_frame_id + "_world", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("PublishTF_" + camera_frame_id + ".inputs:topicName", "/tf"),
                    ("PublishRawTF_" + camera_frame_id + "_world.inputs:topicName", "/tf"),
                    ("PublishRawTF_" + camera_frame_id + "_world.inputs:parentFrameId", camera_frame_id),
                    ("PublishRawTF_" + camera_frame_id + "_world.inputs:childFrameId", camera_frame_id + "_world"),
                    ("PublishRawTF_" + camera_frame_id + "_world.inputs:rotation", [0.5, -0.5, 0.5, 0.5]),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        ros_camera_graph_path + "/OnTick.outputs:tick",
                        "PublishTF_" + camera_frame_id + ".inputs:execIn",
                    ),
                    (
                        ros_camera_graph_path + "/OnTick.outputs:tick",
                        "PublishRawTF_" + camera_frame_id + "_world.inputs:execIn",
                    ),
                    (
                        ros_camera_graph_path + "/IsaacClock.outputs:simulationTime",
                        "PublishTF_" + camera_frame_id + ".inputs:timeStamp",
                    ),
                    (
                        ros_camera_graph_path + "/IsaacClock.outputs:simulationTime",
                        "PublishRawTF_" + camera_frame_id + "_world.inputs:timeStamp",
                    ),
                ],
            },
        )
    except Exception as e:
        print(e)

    set_target_prims(
        primPath=ros_camera_graph_path + "/PublishTF_" + camera_frame_id,
        inputName="inputs:targetPrims",
        targetPrimPaths=[camera_prim],
    )
    return


def setup_camera():
    camera = Camera(
        prim_path=EXISTING_CAMERA_PATH,
        frequency=30,
        resolution=(1280, 720),
    )
    camera.initialize()
    return camera


def setup_camera_publishers(camera, freq=30, context=None):
    publish_camera_tf(camera)
    publish_camera_info(camera, freq)
    publish_rgb(camera, freq)
    publish_depth(camera, freq)
    publish_pointcloud_from_depth(camera, freq)
    setup_camera_pose_publisher(camera, context=context)
