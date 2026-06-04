#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import tf_transformations
from geometry_msgs.msg import PoseStamped
from moveit_planning_service.action import MoveToPose

class MoveToPoseClient(Node):

    def __init__(self):
        super().__init__('test_move_to_pose_client')
        self._action_client = ActionClient(self, MoveToPose, 'move_to_pose')

    def send_goal(self, pose):
        goal_msg = MoveToPose.Goal()
        goal_msg.target_pose = pose

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info('Sending goal request...')
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: success={result.success}, message={result.message}')
        # rclpy.shutdown() # 如果只想执行一次，可以在这里关闭

    def feedback_callback(self, feedback_msg):
        # 如果 Action 定义了 feedback，这里会收到
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback}')

def main(args=None):
    rclpy.init(args=args)
    action_client = MoveToPoseClient()

    # --- 目标一 ---
    pose1 = PoseStamped()
    pose1.header.frame_id = "panda_link0"
    pose1.pose.position.x = 0.4
    pose1.pose.position.y = 0.4
    pose1.pose.position.z = 0.5
    q = tf_transformations.quaternion_from_euler(-1.57, 0, -1.57) 
    pose1.pose.orientation.x = q[0]
    pose1.pose.orientation.y = q[1]
    pose1.pose.orientation.z = q[2]
    pose1.pose.orientation.w = q[3]
    
    action_client.get_logger().info("Sending Goal 1...")
    action_client.send_goal(pose1)
    
    # 使用 spin 来等待回调完成
    # 这里是一个简化的例子，实际应用中可能需要更复杂的逻辑来处理多个目标
    rclpy.spin(action_client)
    
    action_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
