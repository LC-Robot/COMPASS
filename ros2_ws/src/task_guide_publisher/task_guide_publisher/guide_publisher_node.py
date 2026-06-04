#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseArray, Pose

class ManualGuidePublisher(Node):
    def __init__(self):
        # ROS 1: rospy.init_node('manual_guide_publisher')
        # ROS 2: super().__init__('node_name')
        super().__init__('manual_guide_publisher')

        # --- 1. 从ROS参数服务器加载目标坐标 ---
        # ROS 1: rospy.get_param("~target/x", ...)
        # ROS 2: self.declare_parameter("param.name", ...)
        self.declare_parameter("target.x", 0.5)
        self.declare_parameter("target.y", 0.0)
        self.declare_parameter("target.z", 0.05)
        self.declare_parameter("world_frame", "panda_link0")
        self.declare_parameter("publish_rate", 1.0)

        self.target_x = self.get_parameter("target.x").get_parameter_value().double_value
        self.target_y = self.get_parameter("target.y").get_parameter_value().double_value
        self.target_z = self.get_parameter("target.z").get_parameter_value().double_value
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value

        # --- 2. 创建发布者 ---
        # ROS 1的 latch=True 在 ROS 2 中通过QoS实现
        # DurabilityPolicy.TRANSIENT_LOCAL 保证后加入的订阅者也能收到最后发布的消息
        latched_qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        
        # ROS 1: rospy.Publisher(...)
        # ROS 2: self.create_publisher(...)
        self.target_pub = self.create_publisher(
            PoseArray, 
            '/potential_targets', 
            latched_qos_profile
        )
        
        # --- 3. 构建要发布的消息 ---
        self.pose_array_msg = PoseArray()
        self.pose_array_msg.header.frame_id = self.world_frame
        
        heuristic_pose = Pose()
        heuristic_pose.position.x = self.target_x
        heuristic_pose.position.y = self.target_y
        heuristic_pose.position.z = self.target_z
        heuristic_pose.orientation.w = 1.0
        
        self.pose_array_msg.poses.append(heuristic_pose)

        # --- 4. 创建一个定时器来周期性地发布消息 ---
        # ROS 1: rospy.Timer(...)
        # ROS 2: self.create_timer(...)
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_timer_callback)
        
        self.get_logger().info("\033[1;32mManual Guide Publisher is running.\033[0m")
        self.get_logger().info(f"Continuously publishing heuristic target at "
                               f"[{self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}] "
                               f"in frame '{self.world_frame}'")

    def publish_timer_callback(self):
        """定时器回调，用于发布消息"""
        # ROS 1: rospy.Time.now()
        # ROS 2: self.get_clock().now().to_msg()
        self.pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        self.target_pub.publish(self.pose_array_msg)
        self.get_logger().debug("Published heuristic target.")


def main(args=None):
    rclpy.init(args=args)
    node = ManualGuidePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()