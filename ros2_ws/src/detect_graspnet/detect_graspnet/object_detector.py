#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, Bool
from cv_bridge import CvBridge
from collections import deque
import threading
import cv2
import numpy as np
import torch
import os # --- MODIFICATION START ---: 导入os模块以处理文件路径
import time # --- MODIFICATION START ---: 导入time模块以创建唯一文件名

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

        # YOLO-World stores text/class features outside the normal module
        # parameters in some Ultralytics versions. Move those tensors explicitly
        # so CUDA inference does not mix model tensors with CPU text features.
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
        self.PROCESS_TIMER_PERIOD = 0.5 # 增加处理频率以更快地追踪
        
        # --- MODIFICATION START ---: 定义图片保存路径
        self.save_path = os.environ.get(
            'COMPASS_DETECTION_OUTPUT_DIR',
            os.path.join(os.getcwd(), 'detect_images')
        )
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            self.get_logger().info(f"Created directory for saving images at: {self.save_path}")
        # --- MODIFICATION END ---
        
        self.STATE_IDLE = 0      # 模型未加载，暂停
        self.STATE_DETECTING = 1 # 模型已加载，正在寻找第一个目标
        self.STATE_TRACKING = 2  # 已找到目标，正在寻找最佳帧
        self.current_state = self.STATE_IDLE

        # self.tracking_candidates现在将存储包含图像本身的字典
        self.tracking_candidates = [] 
        self.tracking_miss_counter = 0
        self.TRACKING_MISS_THRESHOLD = 3 # 连续丢失3次则认为追踪结束

        self.detector = None
        self.data_queue = deque(maxlen=self.QUEUE_MAX_SIZE)
        self.lock = threading.Lock()

        # --- ROS 2 通信 ---
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
            
            # 清理所有状态
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
        # 只有在 DETECTING 或 TRACKING 状态下才接收数据
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
            # --- MODIFICATION START ---: 获取当前帧的图像
            corresponding_image = images_to_process[i]
            # --- MODIFICATION END ---
            image_center = np.array([corresponding_image.shape[1] / 2, corresponding_image.shape[0] / 2])
            
            # 检查当前帧是否检测到目标
            if detections:
                # 只关心第一个（通常是置信度最高的）检测结果
                best_box = detections[0]['xyxy']
                box_center = np.array([(best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2])
                distance_to_center = np.linalg.norm(box_center - image_center)
                
                # --- MODIFICATION START ---: 创建包含图像的候选字典
                candidate_data = {
                    'distance': distance_to_center,
                    'pose': corresponding_pose,
                    'image': corresponding_image # 将图像也存进去
                }
                # --- MODIFICATION END ---

                # 如果是 DETECTING 状态，第一次检测到就切换到 TRACKING
                if self.current_state == self.STATE_DETECTING:
                    self.get_logger().info(f"\033[1;33mInitial object detected! Switching to TRACKING state.\033[0m")
                    self.current_state = self.STATE_TRACKING
                    self.tracking_candidates.append(candidate_data) # 保存包含图像的字典
                    self.tracking_miss_counter = 0
                
                # 如果是 TRACKING 状态，继续添加候选帧
                elif self.current_state == self.STATE_TRACKING:
                    self.get_logger().info(f"Tracking object... Distance to center: {distance_to_center:.2f}")
                    self.tracking_candidates.append(candidate_data) # 保存包含图像的字典
                    self.tracking_miss_counter = 0

            else: # 如果当前帧没有检测到目标
                if self.current_state == self.STATE_TRACKING:
                    self.tracking_miss_counter += 1
                    self.get_logger().warn(f"Tracking miss... Counter: {self.tracking_miss_counter}/{self.TRACKING_MISS_THRESHOLD}")
                    # 如果连续丢失次数达到阈值，则结束追踪
                    if self.tracking_miss_counter >= self.TRACKING_MISS_THRESHOLD:
                        self.get_logger().info("\033[1;32mTracking lost. Evaluating best frame from candidates...\033[0m")
                        self.evaluate_and_publish_best()
                        return # 评估后直接返回，不再处理本批次剩余图像
    
    def evaluate_and_publish_best(self):
        """从追踪到的候选帧中，选择最佳的一帧并发布。"""
        if not self.tracking_candidates:
            self.get_logger().error("Evaluation called, but no candidates were found. Something went wrong.")
            self.unload_detector() # 即使出错也要清理
            return

        # 找到距离中心最近的那个候选帧
        best_candidate = min(self.tracking_candidates, key=lambda x: x['distance'])
        
        self.get_logger().info(f"Found best frame with distance {best_candidate['distance']:.2f} to center.")
        
        # --- MODIFICATION START ---: 保存最佳图像到本地文件
        try:
            best_image = best_candidate['image']
            # 使用时间戳创建一个唯一的文件名，避免覆盖
            timestamp = int(time.time())
            filename = f"best_detection_image_{timestamp}.png"
            full_path = os.path.join(self.save_path, filename)
            
            cv2.imwrite(full_path, best_image)
            self.get_logger().info(f"\033[1;32mSuccessfully saved best image to: {full_path}\033[0m")
        except Exception as e:
            self.get_logger().error(f"Failed to save the best image: {e}")
        # --- MODIFICATION END ---

        self.get_logger().info("Publishing result with the best synchronized pose.")
        
        # 使用最佳帧的位姿发布结果
        self.publish_detection_result(True, best_candidate['pose'])
        
        # 完成任务，卸载模型并回到IDLE状态
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
