#!/usr/bin/env python3

import rclpy
import time
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_planning_service.action import MoveToJoints


class MoveToJointsClient(Node):

    def __init__(self):
        super().__init__("test_joint_move_client")
        self._action_client = ActionClient(self, MoveToJoints, "move_to_joints")

    def send_goal_and_wait(self, joint_state):
        goal_msg = MoveToJoints.Goal()
        goal_msg.target_joints = joint_state

        self.get_logger().info("Waiting for action server...")
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Action server not available after waiting!")
            return None

        self.get_logger().info("Sending goal request...")
        send_goal_future = self._action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected.")
            return None
        
        self.get_logger().info("Goal accepted.")

        get_result_future = goal_handle.get_result_async()
        self.get_logger().info("Waiting for result...")
        rclpy.spin_until_future_complete(self, get_result_future)

        result = get_result_future.result().result
        self.get_logger().info(f"Result: success={result.success}, message={result.message}")
        return result


def main(args=None):
    rclpy.init(args=args)
    action_client = MoveToJointsClient()

    target_positions = [
        [-0.5, 0.2, 0.2, -1.5, 0.0, 1.5, 0.0],
        [0.5, 0.5, 0.0, -1.0, 0.0, 1.5, 0.785],
        [0.8, -0.5, -0.5, -2.0, 0.0, 1.5, 0.0],
    ]

    for i, positions in enumerate(target_positions):
        action_client.get_logger().info(f"--- Sending Goal #{i + 1} ---")
        
        target_joint_state = JointState()
        target_joint_state.name = [f"panda_joint{j + 1}" for j in range(7)]
        target_joint_state.position = positions

        result = action_client.send_goal_and_wait(target_joint_state)

        if result is None or not result.success:
            action_client.get_logger().error("Action failed or was rejected. Aborting sequence.")
            break
        
        if i < len(target_positions) - 1:
            action_client.get_logger().info("Waiting for 15 seconds before next goal...")
            time.sleep(15)

    action_client.get_logger().info("All goals processed. Shutting down.")
    action_client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
