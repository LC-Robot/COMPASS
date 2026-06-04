#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray, Bool
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import message_filters
import numpy as np
import threading

import spatialmath as sm
from scipy.spatial.transform import Rotation as R

import os
import sys
import open3d as o3d
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))

from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo as GraspNetCameraInfo 
from data_utils import create_point_cloud_from_depth_image

import cv2 
from cv_process import segment_image

def get_net():
    """Load the pretrained GraspNet model."""
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    checkpoint_path = os.path.join(ROOT_DIR, 'logs', 'log_rs', 'checkpoint-rs.tar') 
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print("GraspNet model loaded successfully.")
    return net

def get_and_process_data(color_image, depth_image, mask_image, intrinsic_matrix):
    """Convert RGB-D input and intrinsics into GraspNet input tensors."""
    num_point = 30000
    color = color_image.astype(np.float32) / 255.0
    depth = depth_image.astype(np.float32)
    workspace_mask = mask_image
    height, width, _ = color.shape
    fx, fy, cx, cy = intrinsic_matrix[0,0], intrinsic_matrix[1,1], intrinsic_matrix[0,2], intrinsic_matrix[1,2]
    
    camera_info = GraspNetCameraInfo(width, height, fx, fy, cx, cy, 1.0)
    
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
    mask = (workspace_mask > 0) & (depth > 0) & (depth < 1.5)
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    
    if len(cloud_masked) < 100:
        print("Warning: Not enough points in point cloud after filtering.")
        return None, None
        
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=True)
        
    cloud_sampled = cloud_masked[idxs]
    
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    o3d_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    
    end_points = dict()
    cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    end_points['point_clouds'] = cloud_sampled_tensor.to(device)
    
    return end_points, o3d_cloud

def get_grasps(net, end_points):
    """Run GraspNet inference."""
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud_points):
    """Filter grasp candidates with model-free collision detection."""
    mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=0.01)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
    return gg[~collision_mask]


class GraspNetROSNode(Node):
    def __init__(self, net):
        super().__init__('graspnet_inference_node')
        self.net = net

        self.bridge = CvBridge()
        self.lock = threading.Lock() 
        self.processing = False 
        
        self.active = False
        self.trigger_lock = threading.Lock()
        
        self.publishing_timer = None
        self.last_successful_grasp_msg = None
        self.publisher_rate = 30.0
        
        rgb_topic = '/camera_rgb'
        depth_topic = '/camera_depth'
        cam_info_topic = '/camera_camera_info'
        camera_pose_topic = '/camera_pose'

        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.cam_info_sub = message_filters.Subscriber(self, CameraInfo, cam_info_topic)
        self.camera_pose_sub = message_filters.Subscriber(self, PoseStamped, camera_pose_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.cam_info_sub, self.camera_pose_sub], 
            queue_size=10,
            slop=0.2
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        self.trigger_subscriber = self.create_subscription(
            Bool,
            '/trigger_graspnet',
            self.trigger_callback,
            10
        )

        self.grasp_publisher = self.create_publisher(Float64MultiArray, '/grasp', 10)
        self.get_logger().info('GraspNet ROS2 node started.')
        self.get_logger().info('Waiting for grasp trigger on /trigger_graspnet topic...')

    def trigger_callback(self, msg):
        """Activate one-frame GraspNet processing from the trigger topic."""
        if self.publishing_timer is not None:
            self.publishing_timer.cancel()
            self.publishing_timer = None
            self.last_successful_grasp_msg = None
            self.get_logger().info("Stopped continuous grasp publishing due to new trigger.")
            
        with self.trigger_lock:
            if msg.data:
                if not self.active:
                    self.get_logger().info("\033[1;32mReceived ACTIVATION trigger. Will process next synchronized frame.\033[0m")
                    self.active = True
            else:
                if self.active:
                    self.get_logger().info("Received DEACTIVATION trigger.")
                    self.active = False

    def publish_loop_callback(self):
        """Republish the last successful grasp result at a fixed rate."""
        if self.last_successful_grasp_msg is not None:
            self.grasp_publisher.publish(self.last_successful_grasp_msg)
            self.get_logger().info("Continuously publishing grasp pose...", throttle_duration_sec=1.0)

    def synchronized_callback(self, rgb_msg, depth_msg, cam_info_msg, camera_pose_msg):
        with self.trigger_lock:
            if not self.active:
                return
        
        with self.lock:
            if self.processing:
                self.get_logger().warn('Still processing previous frame, skipping new one.', throttle_duration_sec=1)
                return
            self.processing = True
            with self.trigger_lock:
                self.active = False

        self.get_logger().info('Node is ACTIVE. Received a synchronized frame. Starting processing...')
        
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            intrinsic_matrix = np.array(cam_info_msg.k).reshape(3, 3)
            camera_position = [camera_pose_msg.pose.position.x, camera_pose_msg.pose.position.y, camera_pose_msg.pose.position.z]
            camera_orientation = [camera_pose_msg.pose.orientation.w, camera_pose_msg.pose.orientation.x, camera_pose_msg.pose.orientation.y, camera_pose_msg.pose.orientation.z]
            
            self.run_graspnet_pipeline(rgb_image, depth_image, intrinsic_matrix, camera_position, camera_orientation)

        except Exception as e:
            self.get_logger().error(f"An error occurred during processing: {e}")
        finally:
            with self.lock:
                self.processing = False

    def publish_grasp_result(self, detect_flag, grasps_with_scores=None):
        """Publish grasp candidates as a flat Float64MultiArray."""
        msg = Float64MultiArray()
        data = []
        
        if detect_flag and grasps_with_scores:
            num_grasps = len(grasps_with_scores)
            data.append(1.0) 
            data.append(float(num_grasps))
            
            for grasp_info in grasps_with_scores:
                data.extend(grasp_info['position'].tolist())
                data.extend(grasp_info['rotation'].tolist())
                data.append(float(grasp_info['score']))
            
            msg.data = data
            
            self.last_successful_grasp_msg = msg
            self.grasp_publisher.publish(self.last_successful_grasp_msg)
            self.get_logger().info(f'Published FIRST successful result with {num_grasps} grasps. Starting continuous publishing...')

            if self.publishing_timer is not None:
                self.publishing_timer.cancel()
            self.publishing_timer = self.create_timer(1.0 / self.publisher_rate, self.publish_loop_callback)
        else: 
            data.append(0.0)
            msg.data = data
            self.last_successful_grasp_msg = None
            if self.publishing_timer is not None:
                self.publishing_timer.cancel()
                self.publishing_timer = None
            self.grasp_publisher.publish(msg)
            self.get_logger().info('Published FAILURE grasp result (no valid grasp found).')

    def run_graspnet_pipeline(self, rgb, depth, intrinsics, camera_position=None, camera_orientation=None):
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        mask = segment_image(rgb_bgr)
        if np.count_nonzero(mask) < 100:
            self.get_logger().warn("Segmentation resulted in a very small mask. Aborting.")
            self.publish_grasp_result(False)
            return

        end_points, cloud = get_and_process_data(rgb, depth, mask, intrinsics)
        if end_points is None or cloud is None:
            self.get_logger().error("Failed to process data into a valid point cloud.")
            self.publish_grasp_result(False)
            return

        gg = get_grasps(self.net, end_points)
        if len(gg) == 0:
            self.get_logger().warn("GraspNet inference did not produce any grasp candidates.")
            self.publish_grasp_result(False)
            return

        gg = collision_detection(gg, np.array(cloud.points))
        if len(gg) == 0:
            self.get_logger().warn("No grasps remained after collision detection.")
            self.publish_grasp_result(False)
            return
            
        self.get_logger().info("Angle filtering is disabled. Passing all grasps after collision detection.")
        filtered_grasps = list(gg)
        
        if not filtered_grasps:
            self.get_logger().warn("No grasps available after collision detection.")
            self.publish_grasp_result(False)
            return

        filtered_grasps.sort(key=lambda g: g.score, reverse=True)
        
        num_to_visualize_and_send = 5
        top_n_grasps = filtered_grasps[:num_to_visualize_and_send]
        
        self.get_logger().info(f"Found {len(top_n_grasps)} suitable grasps to visualize and send.")

        if top_n_grasps:
            self.get_logger().info(f"--- Visualizing top {len(top_n_grasps)} grasp(s) ---")
            grippers = [g.to_open3d_geometry() for g in top_n_grasps]
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([cloud, coordinate_frame, *grippers])

        world_grasps_to_send = []
        if camera_position is not None and camera_orientation is not None:
            for grasp in top_n_grasps:
                pos, rot = self.transform_camera_to_world(
                    grasp.translation, grasp.rotation_matrix, 
                    camera_position, camera_orientation
                )
                if pos is not None and rot is not None:
                    world_grasps_to_send.append({
                        'position': pos, 
                        'rotation': rot,
                        'score': grasp.score
                    })
            
            if not world_grasps_to_send:
                self.get_logger().error("Coordinate transformation failed for all selected grasps.")
                self.publish_grasp_result(False)
                return
        else:
            self.get_logger().warn("No camera pose provided. Cannot transform grasps to world frame or publish.")
            self.publish_grasp_result(False)
            return

        self.get_logger().info("--- Publishing successful grasp results (with scores) ---")
        self.publish_grasp_result(True, world_grasps_to_send)

    def transform_camera_to_world(self, translation, rotation, camera_position, camera_orientation):
        try:
            T_co = sm.SE3.Trans(translation) * sm.SE3(sm.SO3.TwoVectors(x=rotation[:, 0], y=rotation[:, 1]))
            camera_world_rotation = R.from_quat([camera_orientation[1], camera_orientation[2], camera_orientation[3], camera_orientation[0]]).as_matrix()
            T_wc = sm.SE3.Trans(camera_position) * sm.SE3(sm.SO3.TwoVectors(x=camera_world_rotation[:, 0], y=camera_world_rotation[:, 1]))
            T_wo = T_wc * T_co
            return self.extract_position_and_orientation_from_transform(T_wo.A)
        except Exception as e:
            self.get_logger().error(f"Error transforming camera to world: {e}")
            return None, None

    def extract_position_and_orientation_from_transform(self, T):
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        rot = R.from_matrix(rotation_matrix)
        quat_xyzw = rot.as_quat()
        orientation = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return position, orientation

def main(args=None):
    rclpy.init(args=args)
    try:
        net = get_net()
        graspnet_node = GraspNetROSNode(net)
        rclpy.spin(graspnet_node)
    except Exception as e:
        print(f"An error occurred in main: {e}")
    finally:
        if 'graspnet_node' in locals() and rclpy.ok():
            graspnet_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
