#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, Bool
from cv_bridge import CvBridge
from collections import deque
import threading
import numpy as np
import torch
import os

import message_filters

from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8s-world.pt"):
        self.device = os.environ.get(
            "COMPASS_YOLO_DEVICE",
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"YOLOv8-world model loaded from {model_path} on {self.device}")

    def set_target_class(self, target_class):
        self.model.set_classes([target_class])

        model = getattr(self.model, "model", None)
        if model is None:
            return
        for attr_name in ("txt_feats", "text_feats", "clip_feats"):
            attr_value = getattr(model, attr_name, None)
            if torch.is_tensor(attr_value):
                setattr(model, attr_name, attr_value.to(self.device))

    def detect_batch(self, images_bgr, target_class):
        self.set_target_class(target_class)
        results = self.model.predict(images_bgr, verbose=False, device=self.device)
        batch_detections = []
        for result in results:
            boxes = result.boxes
            current_image_detections = []
            if boxes:
                for box in boxes:
                    if box.conf.item() > 0.25:
                        current_image_detections.append({
                            "xyxy": box.xyxy[0].tolist(),
                            "conf": box.conf.item(),
                            "cls": result.names[int(box.cls.item())]
                        })
            batch_detections.append(current_image_detections)
        return batch_detections


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector_node')

        self.QUEUE_MAX_SIZE = 30
        self.TARGET_CLASS = "banana"
        self.PROCESS_TIMER_PERIOD = 0.5
        
        self.STATE_IDLE = 0
        self.STATE_DETECTING = 1
        self.STATE_TRACKING = 2
        self.current_state = self.STATE_IDLE

        self.tracking_candidates = [] 
        self.tracking_miss_counter = 0
        self.TRACKING_MISS_THRESHOLD = 3

        self.detector = None
        self.data_queue = deque(maxlen=self.QUEUE_MAX_SIZE)
        self.lock = threading.Lock()

        self.bridge = CvBridge()
        self.detection_publisher = self.create_publisher(Float64MultiArray, '/detection_result', 10)
        self.resume_subscriber = self.create_subscription(
            Bool,
            '/resume_detection',
            self.resume_callback,
            10)

        rgb_sub = message_filters.Subscriber(self, Image, '/camera_rgb')
        pose_sub = message_filters.Subscriber(self, PoseStamped, '/camera_pose')
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, pose_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synchronized_callback)
        
        self.process_timer = self.create_timer(self.PROCESS_TIMER_PERIOD, self.process_batch)

        self.get_logger().info(f"Object Detector Node started. Target class: '{self.TARGET_CLASS}'")
        self.get_logger().info("Node is in IDLE state. Waiting for '/resume_detection' signal...")

    def load_detector(self):
        if self.detector is None:
            self.get_logger().info("Loading YOLOv8-world model into VRAM...")
            self.detector = ObjectDetector()
            self.current_state = self.STATE_DETECTING
            self.get_logger().info("Model loaded. State changed to DETECTING.")
        else:
            self.get_logger().warn("Detector is already loaded.")

    def unload_detector(self):
        if self.detector is not None:
            self.get_logger().info("Unloading YOLOv8-world model and releasing VRAM...")
            del self.detector.model
            del self.detector
            self.detector = None
            torch.cuda.empty_cache()
            
            with self.lock:
                self.data_queue.clear()
            self.tracking_candidates = []
            self.tracking_miss_counter = 0
            self.current_state = self.STATE_IDLE
            
            self.get_logger().info("Model unloaded and VRAM released. State changed to IDLE.")
        else:
            self.get_logger().warn("Detector is not loaded, nothing to unload.")

    def resume_callback(self, msg):
        if msg.data:
            self.load_detector()
        else:
            self.unload_detector()
            
    def synchronized_callback(self, rgb_msg, pose_msg):
        if self.current_state == self.STATE_IDLE:
            self.get_logger().debug("State is IDLE. Skipping frame.", throttle_duration_sec=5)
            return

        with self.lock:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                self.data_queue.append({'image': cv_image, 'pose': pose_msg})
            except Exception as e:
                self.get_logger().error(f"CvBridge Error in synchronized_callback: {e}")

    def process_batch(self):
        if self.current_state == self.STATE_IDLE:
            return

        images_to_process, poses_to_process = [], []
        with self.lock:
            if not self.data_queue: return
            while self.data_queue:
                data_pair = self.data_queue.popleft()
                images_to_process.append(data_pair['image'])
                poses_to_process.append(data_pair['pose'])
        if not images_to_process: return

        self.get_logger().info(f"Processing a batch of {len(images_to_process)} images in state {self.current_state}...")
        
        try:
            batch_detections = self.detector.detect_batch(images_to_process, self.TARGET_CLASS)
        except Exception as e:
            self.get_logger().error(f"Error during batch detection: {e}")
            return

        for i, detections in enumerate(batch_detections):
            corresponding_pose = poses_to_process[i]
            corresponding_image = images_to_process[i]
            image_center = np.array([corresponding_image.shape[1] / 2, corresponding_image.shape[0] / 2])
            
            if detections:
                best_box = detections[0]['xyxy']
                box_center = np.array([(best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2])
                distance_to_center = np.linalg.norm(box_center - image_center)
                
                candidate_data = {
                    'distance': distance_to_center,
                    'pose': corresponding_pose,
                }

                if self.current_state == self.STATE_DETECTING:
                    self.get_logger().info("\033[1;33mInitial object detected! Switching to TRACKING state.\033[0m")
                    self.current_state = self.STATE_TRACKING
                    self.tracking_candidates.append(candidate_data)
                    self.tracking_miss_counter = 0
                
                elif self.current_state == self.STATE_TRACKING:
                    self.get_logger().info(f"Tracking object... Distance to center: {distance_to_center:.2f}")
                    self.tracking_candidates.append(candidate_data)
                    self.tracking_miss_counter = 0

            else:
                if self.current_state == self.STATE_TRACKING:
                    self.tracking_miss_counter += 1
                    self.get_logger().warn(f"Tracking miss... Counter: {self.tracking_miss_counter}/{self.TRACKING_MISS_THRESHOLD}")
                    if self.tracking_miss_counter >= self.TRACKING_MISS_THRESHOLD:
                        self.get_logger().info("\033[1;32mTracking lost. Evaluating best frame from candidates...\033[0m")
                        self.evaluate_and_publish_best()
                        return
    
    def evaluate_and_publish_best(self):
        """Publish the pose synchronized with the best tracked frame."""
        if not self.tracking_candidates:
            self.get_logger().error("Evaluation called, but no candidates were found. Something went wrong.")
            self.unload_detector()
            return

        best_candidate = min(self.tracking_candidates, key=lambda x: x['distance'])
        
        self.get_logger().info(f"Found best frame with distance {best_candidate['distance']:.2f} to center.")
        self.get_logger().info("Publishing result with the best synchronized pose.")
        self.publish_detection_result(True, best_candidate['pose'])
        self.unload_detector()

    def publish_detection_result(self, detected_flag, pose_msg):
        msg = Float64MultiArray()
        data = []
        if detected_flag and pose_msg is not None:
            data.append(1.0)
            p = pose_msg.pose.position
            o = pose_msg.pose.orientation
            data.extend([p.x, p.y, p.z, o.w, o.x, o.y, o.z])
            self.get_logger().info(f"Published detection at pose: pos(x={p.x:.3f}, y={p.y:.3f}, z={p.z:.3f})")
        else:
            data.append(0.0)
            self.get_logger().warn("Publishing failure signal (no detection).")
        msg.data = data
        self.detection_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    try:
        object_detector_node = ObjectDetectorNode()
        rclpy.spin(object_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'object_detector_node' in locals() and rclpy.ok():
            object_detector_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
