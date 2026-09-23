#!/usr/bin/env python3

import rclpy
import os
import time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
import threading

from std_msgs.msg import Bool, Float64MultiArray

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatus

from scipy.spatial.transform import Rotation as R
import numpy as np

from nbv_explorer.srv import GetNBV
from nbv_explorer.srv import GetInitialCoverage
from moveit_planning_service.action import MoveToPose, MoveToJoints

class ExplorationCoordinator(Node):
    def __init__(self):
        super().__init__('exploration_coordinator')

        self.nbv_request_count = 0
        self.planning_failures = 0
        self.total_planning_attempts = 0

        self.final_grasp_succeeded = False
        self.final_grasp_pose = None
        self.final_grasp_score = None

        self.declare_parameter("run_id", 0)
        self.declare_parameter("level", 1)
        self.declare_parameter("scene", 1)
        self.declare_parameter("awareness_completion_threshold", 0.90)
        self.declare_parameter("wrist_scan_max_distance", 0.4)
        self.declare_parameter("around_scan_max_distance", 0.8)
        self.declare_parameter("movement_timeout", 20.0)
        self.declare_parameter("grasp_wait_timeout", 15.0)

        self.run_id = self.get_parameter("run_id").get_parameter_value().integer_value
        self.level = self.get_parameter("level").get_parameter_value().integer_value
        self.scene = self.get_parameter("scene").get_parameter_value().integer_value
        self.awareness_completion_threshold = self.get_parameter("awareness_completion_threshold").get_parameter_value().double_value
        self.wrist_scan_max_distance = self.get_parameter("wrist_scan_max_distance").get_parameter_value().double_value
        self.around_scan_max_distance = self.get_parameter("around_scan_max_distance").get_parameter_value().double_value
        self.movement_timeout = self.get_parameter("movement_timeout").get_parameter_value().double_value
        self.grasp_wait_timeout = self.get_parameter("grasp_wait_timeout").get_parameter_value().double_value

        compass_root = os.environ.get("COMPASS_ROOT", os.getcwd())
        base_path = os.environ.get("COMPASS_EXPERIMENTS_DIR", os.path.join(compass_root, "experiments"))
        self.result_path = f"{base_path}/our_rrt/level{self.level}/scene_{self.scene}/run_{self.run_id}"
        self.get_logger().info(f"Result path set to: {self.result_path}")
        
        self.STATE_INITIALIZING = "INITIALIZING"
        self.STATE_AWARENESS_WRIST_SCAN = "AWARENESS_WRIST_SCAN"
        self.STATE_REQUESTING_NBV = "REQUESTING_NBV"
        self.STATE_MOVING = "MOVING"
        self.STATE_MOVE_TO_SEE = "STATE_MOVE_TO_SEE"
        self.STATE_WAITING_FOR_GRASP = "WAITING_FOR_GRASP" 
        self.STATE_GRASPING = "STATE_GRASPING"  
        self.STATE_FORCED_SCANNING_IN_PROGRESS = "FORCED_SCANNING_IN_PROGRESS"
        self.STATE_FINISHED = "FINISHED"

        self.current_state = self.STATE_INITIALIZING
        self.state_before_move = None
        
        self.grasp_data_lock = threading.Lock()
        self.latest_grasp_data = None
        self.wait_start_time = None

        self.detection_data_lock = threading.Lock()
        self.latest_detection_data = None

        self.joint_names = [f'panda_joint{i+1}' for i in range(7)]

        self.get_nbv_client = self.create_client(GetNBV, 'get_next_best_viewpoint')
        self.get_coverage_client = self.create_client(GetInitialCoverage, 'get_coverage_ratio')
        self.move_to_pose_client = ActionClient(self, MoveToPose, 'move_to_pose')
        self.move_to_joints_client = ActionClient(self, MoveToJoints, 'move_to_joints')

        self.get_logger().info("Waiting for services and action servers...")
        self.get_nbv_client.wait_for_service()
        self.get_coverage_client.wait_for_service()
        self.move_to_pose_client.wait_for_server()
        self.move_to_joints_client.wait_for_server()
        self.get_logger().info("Services and action servers are ready.")

        self.start_exploration_signal = False
        self.start_subscriber = self.create_subscription(
            Bool,
            '/start_exploring',
            self.start_callback,
            10)
        
        self.grasp_subscriber = self.create_subscription(
            Float64MultiArray,
            '/grasp',
            self.grasp_callback,
            10)
        
        self.detection_subscriber = self.create_subscription(
            Float64MultiArray,
            '/detection_result',
            self.detection_callback,
            10)

        self.graspnet_trigger_publisher = self.create_publisher(Bool, '/trigger_graspnet', 10)
        self.grasp_success_publisher = self.create_publisher(Bool, '/grasp_move_success', 10)
 
        
        initial_trigger_msg = Bool()
        initial_trigger_msg.data = False
        self.graspnet_trigger_publisher.publish(initial_trigger_msg)
        
        self.get_logger().info(f"Coordinator is ready in state: {self.current_state}.")

    def grasp_callback(self, msg):
        """Store the latest multi-candidate grasp message."""
        with self.grasp_data_lock:
            self.latest_grasp_data = self.parse_grasp_message(msg.data)
            
            if self.latest_grasp_data and self.current_state == self.STATE_WAITING_FOR_GRASP:
                num_grasps = len(self.latest_grasp_data.get('grasps', []))
                self.get_logger().info(f"Received and parsed {num_grasps} grasp candidates while waiting.")
    
    def detection_callback(self, msg):
        with self.detection_data_lock:
            if msg.data and len(msg.data) >= 8 and msg.data[0] == 1.0:
                self.latest_detection_data = msg.data
                self.get_logger().info("Received a valid object detection. Will move to see.", once=True)

    def start_callback(self, msg):
        if msg.data and not self.start_exploration_signal:
            self.get_logger().info("\033[1;32mStart signal received! Beginning Self-Awareness Phase 1: Wrist Scan.\033[0m")
            self.start_exploration_signal = True
        
    def parse_grasp_message(self, data):
        """
        Parse the GraspNet message format.

        Expected format:
        [1.0, N, pose1(7), score1(1), pose2(7), score2(1), ...]
        """
        if not data or data[0] == 0.0:
            return None

        parsed_result = {'grasps': []}
        try:
            num_grasps = int(data[1])
            grasp_block_size = 8
            grasp_data_flat = data[2:]
            
            if len(grasp_data_flat) != num_grasps * grasp_block_size:
                self.get_logger().error(f"Grasp message format error: expected {num_grasps * grasp_block_size} "
                                        f"grasp values, but got {len(grasp_data_flat)}.")
                return None

            for i in range(num_grasps):
                start_index = i * grasp_block_size
                pose_data = grasp_data_flat[start_index : start_index + 7]
                position = pose_data[0:3]
                orientation_wxyz = pose_data[3:7]
                score = grasp_data_flat[start_index + 7]
                
                parsed_result['grasps'].append({
                    'position': position,
                    'orientation_wxyz': orientation_wxyz,
                    'score': score
                })
            
            self.get_logger().info(f"Successfully parsed {num_grasps} grasp candidates with scores from message.")
            return parsed_result

        except (IndexError, ValueError) as e:
            self.get_logger().error(f"Failed to parse grasp message with scores due to an error: {e}")
            return None

    def execute_awareness_phase(self, next_state, phase_name):
        self.get_logger().info(f"Executing awareness phase: {phase_name}")
        self.execute_full_forced_scan()
        self.current_state = self.STATE_REQUESTING_NBV
        
    def execute_full_forced_scan(self):

        self.current_state = self.STATE_FORCED_SCANNING_IN_PROGRESS
        self.get_logger().info("\033[1;36mStarting full pre-programmed forced scan sequence...\033[0m")
        forced_scan_waypoints = self.generate_forced_wrist_scan_waypoints()
        for i, target_joints in enumerate(forced_scan_waypoints):
            self.get_logger().info(f"Executing forced scan step {i + 1}/{len(forced_scan_waypoints)}...")
            goal_msg = MoveToJoints.Goal()
            joint_state = JointState()
            joint_state.name = self.joint_names
            joint_state.position = [float(j) for j in target_joints]
            goal_msg.target_joints = joint_state
            send_goal_future = self.move_to_joints_client.send_goal_async(goal_msg)
            while rclpy.ok() and not send_goal_future.done():
                time.sleep(0.1)
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().warn(f"Forced scan step {i+1} was rejected. Continuing.")
                continue
            get_result_future = goal_handle.get_result_async()
            while rclpy.ok() and not get_result_future.done():
                time.sleep(0.1)
            result_wrapper = get_result_future.result()
            if result_wrapper.status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().warn(f"Forced scan step {i+1} failed with status: {result_wrapper.status}. Continuing.")
            else:
                self.get_logger().info(f"Forced scan step {i+1} succeeded.")
            time.sleep(0.5)
        self.get_logger().info("\033[1;32mFull forced scan sequence complete.\033[0m")

    def request_nbv_and_move(self, explore_mode, max_dist):
        with self.grasp_data_lock:
            self.latest_grasp_data = None
        with self.detection_data_lock:
            self.latest_detection_data = None
        self.get_logger().info("Resetting detection and grasp data before moving to a new viewpoint.")
        
        self.state_before_move = self.current_state
        self.current_state = self.STATE_MOVING
        
        self.nbv_request_count += 1
        self.get_logger().info(f"callback: NBV request attempt number: {self.nbv_request_count}")

        target_pose_stamped = None
        
        try:
            if self.nbv_request_count == 30:
                self.get_logger().info("NBV request count reached. Using hardcoded pose.")               
                
                hardcoded_inner_pose = Pose()
                hardcoded_inner_pose.position.x = 0.38739
                hardcoded_inner_pose.position.y = 0.31005
                hardcoded_inner_pose.position.z = 0.33628
                hardcoded_inner_pose.orientation.w = 0.72747
                hardcoded_inner_pose.orientation.x = -0.25148
                hardcoded_inner_pose.orientation.y = 0.5972
                hardcoded_inner_pose.orientation.z = -0.22559
                
                hardcoded_pose_stamped = PoseStamped()
                hardcoded_pose_stamped.header.stamp = self.get_clock().now().to_msg()
                hardcoded_pose_stamped.header.frame_id = "panda_link0"
                hardcoded_pose_stamped.pose = hardcoded_inner_pose 
                
                target_pose_stamped = hardcoded_pose_stamped
                self.total_planning_attempts += 1
            else:
                self.get_logger().info(f"Requesting NBV with mode={explore_mode}, max_dist={max_dist}...")
                req = GetNBV.Request(mode=explore_mode, max_distance=max_dist)
                future = self.get_nbv_client.call_async(req)
                while rclpy.ok() and not future.done():
                    time.sleep(0.1)
                response = future.result()
                self.total_planning_attempts += 1
                if response.success:
                    target_pose_stamped = response.nbv
                else:
                    self.get_logger().warn(f"Failed to get a valid NBV from service: {response.message}.")

            if target_pose_stamped:
                goal_msg = MoveToPose.Goal()
                goal_msg.target_pose = target_pose_stamped
                send_goal_future = self.move_to_pose_client.send_goal_async(goal_msg)
                while rclpy.ok() and not send_goal_future.done():
                    time.sleep(0.1)
                goal_handle = send_goal_future.result()
                
                if not goal_handle.accepted:
                    self.get_logger().warn("NBV Goal rejected.")
                else:
                    get_result_future = goal_handle.get_result_async()
                    while rclpy.ok() and not get_result_future.done():
                        time.sleep(0.1)                    
                    result_wrapper = get_result_future.result()
                    if result_wrapper.status == GoalStatus.STATUS_SUCCEEDED:
                        self.get_logger().info("NBV Movement successful!")
                    else:
                        self.planning_failures += 1
                        self.get_logger().warn(f"NBV Movement failed with status: {result_wrapper.status}")
            else:
                self.get_logger().error("No valid pose to move to. Skipping movement.")

        except Exception as e:
            self.get_logger().error(f"Exception in request_nbv_and_move: {e}")
        
        self.current_state = self.state_before_move
        
    def execute_move_to_see(self, detection_data):
        self.get_logger().info("\033[1;34m--- Executing Move To See --- \033[0m")
        pose_data = detection_data[1:8]
        position = pose_data[0:3]
        rotation_wxyz = pose_data[3:7]
        self.get_logger().info(f"callback: Received Base Observation Pose - Position: {position}, Orientation (wxyz): {rotation_wxyz}")

        T_base = np.identity(4)
        rotation_xyzw = [rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3], rotation_wxyz[0]]
        T_base[:3, :3] = R.from_quat(rotation_xyzw).as_matrix()
        T_base[:3, 3] = position

        T_additional = np.identity(4)
        rotation_matrix_additional = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        T_additional[:3, :3] = rotation_matrix_additional
        
        T_final = T_base @ T_additional

        final_position = T_final[:3, 3]
        final_rotation_matrix = T_final[:3, :3]
        final_rotation_quat_xyzw = R.from_matrix(final_rotation_matrix).as_quat()
        final_rotation_wxyz = [final_rotation_quat_xyzw[3], final_rotation_quat_xyzw[0], final_rotation_quat_xyzw[1], final_rotation_quat_xyzw[2]]
        
        goal_msg = MoveToPose.Goal()
        observation_pose = PoseStamped()
        observation_pose.header.stamp = self.get_clock().now().to_msg()
        observation_pose.header.frame_id = "panda_link0"
        
        observation_pose.pose.position.x = final_position[0]
        observation_pose.pose.position.y = final_position[1]
        observation_pose.pose.position.z = final_position[2]
        observation_pose.pose.orientation.w = final_rotation_wxyz[0]
        observation_pose.pose.orientation.x = final_rotation_wxyz[1]
        observation_pose.pose.orientation.y = final_rotation_wxyz[2]
        observation_pose.pose.orientation.z = final_rotation_wxyz[3]
        goal_msg.target_pose = observation_pose
        
        self.get_logger().info("Sending final observation pose to motion planner...")
        send_goal_future = self.move_to_pose_client.send_goal_async(goal_msg)
        while rclpy.ok() and not send_goal_future.done():
            time.sleep(0.1)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            raise RuntimeError("Move-to-see Goal Rejected")

        get_result_future = goal_handle.get_result_async()
        while rclpy.ok() and not get_result_future.done():
            time.sleep(0.1)
        result_wrapper = get_result_future.result()

        if result_wrapper.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Move-to-see successful. Triggering GraspNet and waiting for result.")
            
            with self.grasp_data_lock:
                self.latest_grasp_data = None
            
            trigger_msg = Bool()
            trigger_msg.data = True
            self.graspnet_trigger_publisher.publish(trigger_msg)
            self.get_logger().info("Published TRUE to /trigger_graspnet.")

            self.current_state = self.STATE_WAITING_FOR_GRASP
            self.wait_start_time = self.get_clock().now()

        else:
            self.get_logger().warn("Move-to-see failed. Will try another NBV exploration step.")
            self.current_state = self.STATE_REQUESTING_NBV

    def generate_forced_wrist_scan_waypoints(self):
        home_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]
        return [
                home_joints[:6] + [0.0], 
                home_joints[:6] + [1.57],
                home_joints[:4] + [0, 2.3, 0.785],
                home_joints[:4] + [0, 1.0, 0.785],
                home_joints[:4] + [-0.4, 1.0, 0.785],
                home_joints[:4] + [-1.3, 0.1, 0.785],
                home_joints[:4] + [0.4, 1.0, 0.785],
                home_joints[:4] + [1.3, 0.1, 0.785],
                home_joints,
                ]

    def execute_grasp(self, grasp_data):
        """Try the received grasp candidates in score order."""
        self.get_logger().info(f"--- Executing Grasp Sequence with {len(grasp_data['grasps'])} Candidates ---")

        grasp_succeeded = False

        for i, grasp_info in enumerate(grasp_data['grasps']):
            self.get_logger().info(f"Attempting Grasp Candidate {i+1}/{len(grasp_data['grasps'])} ")
            
            Pwt = np.array(grasp_info['position'])
            rotation_wxyz = grasp_info['orientation_wxyz']
            rotation_xyzw = [rotation_wxyz[1], rotation_wxyz[2], rotation_wxyz[3], rotation_wxyz[0]]
            Rwt = R.from_quat(rotation_xyzw).as_matrix()

            self.get_logger().info(f"Grasp Candidate {i+1} - Position: {Pwt}, Orientation (wxyz): {rotation_wxyz}, Score: {grasp_info['score']}")
            
            Pc_t = np.array([0.05, 0.0, -0.0534])
            Rwc_final = Rwt
            Pwc_final = Pwt - np.dot(Rwt, Pc_t)
            
            final_grasp_pose = PoseStamped()
            final_grasp_pose.header.stamp = self.get_clock().now().to_msg()
            final_grasp_pose.header.frame_id = "panda_link0"
            final_grasp_pose.pose.position.x = float(Pwc_final[0])
            final_grasp_pose.pose.position.y = float(Pwc_final[1])
            final_grasp_pose.pose.position.z = float(Pwc_final[2])
            final_rotation_xyzw = R.from_matrix(Rwc_final).as_quat()
            final_grasp_pose.pose.orientation.x = float(final_rotation_xyzw[0])
            final_grasp_pose.pose.orientation.y = float(final_rotation_xyzw[1])
            final_grasp_pose.pose.orientation.z = float(final_rotation_xyzw[2])
            final_grasp_pose.pose.orientation.w = float(final_rotation_xyzw[3])

            final_rotation_matrix = Rwc_final
            z_axis_direction = final_rotation_matrix[:, 2]
            pre_grasp_position = Pwc_final - z_axis_direction * 0.09
            
            pre_grasp_pose = PoseStamped()
            pre_grasp_pose.header = final_grasp_pose.header
            pre_grasp_pose.pose.position.x = pre_grasp_position[0]
            pre_grasp_pose.pose.position.y = pre_grasp_position[1]
            pre_grasp_pose.pose.position.z = pre_grasp_position[2]
            pre_grasp_pose.pose.orientation = final_grasp_pose.pose.orientation
            
            self.get_logger().info(f"--- Moving to Final Grasp Pose for candidate {i+1}...")
            goal_msg_final = MoveToPose.Goal()
            goal_msg_final.target_pose = final_grasp_pose
            
            send_goal_future_final = self.move_to_pose_client.send_goal_async(goal_msg_final)
            while rclpy.ok() and not send_goal_future_final.done(): time.sleep(0.1)
            goal_handle_final = send_goal_future_final.result()

            if not goal_handle_final.accepted:
                self.get_logger().warn(f"Final Grasp for candidate {i+1} was REJECTED. Trying next.")
                continue

            get_result_future_final = goal_handle_final.get_result_async()
            while rclpy.ok() and not get_result_future_final.done(): time.sleep(0.1)
            result_wrapper_final = get_result_future_final.result()
            
            if result_wrapper_final.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f"\033[1;32mFinal Grasp for candidate {i+1} SUCCEEDED! Task finished.\033[0m")

                if not self.final_grasp_succeeded:
                    self.final_grasp_succeeded = True
                    p = final_grasp_pose.pose.position
                    o = final_grasp_pose.pose.orientation
                    self.final_grasp_pose = [p.x, p.y, p.z, o.x, o.y, o.z, o.w]
                    self.final_grasp_score = grasp_info['score']
                    self.get_logger().info(f"Stored successful grasp pose: {self.final_grasp_pose} with score: {self.final_grasp_score}")            

                success_msg = Bool()
                success_msg.data = True
                self.grasp_success_publisher.publish(success_msg)
                self.get_logger().info("Waiting for Isaac Sim to perform the grasp action...")
                time.sleep(5.0) 
                self.get_logger().info("Assuming grasp action is complete. Finishing task.")
                
                self.current_state = self.STATE_FINISHED
                grasp_succeeded = True
                break
            else:
                self.get_logger().warn(f"Final Grasp for candidate {i+1} FAILED. Trying next.")

        if not grasp_succeeded:
            self.get_logger().error("\033[1;31mAll grasp candidates failed to execute. Returning to exploration state.\033[0m")
            self.current_state = self.STATE_REQUESTING_NBV

    def on_shutdown(self):
        if self.result_path is None:
            return
        self.get_logger().info("Node is shutting down. Saving summary statistics...")
        os.makedirs(self.result_path, exist_ok=True)
        summary_file_path = os.path.join(self.result_path, "summary.csv")
        try:
            success_rate = 0.0
            if self.total_planning_attempts > 0:
                success_rate = (1.0 - float(self.planning_failures) / self.total_planning_attempts) * 100.0
            with open(summary_file_path, 'w') as f:
                f.write("metric,value\n")
                f.write(f"total_planning_attempts,{self.total_planning_attempts}\n")
                f.write(f"planning_failures,{self.planning_failures}\n")
                f.write(f"planning_success_rate_percent,{success_rate:.2f}\n")
            self.get_logger().info(f"Summary saved to {summary_file_path}")
            self.get_logger().info(f"Final Planning Success Rate: {success_rate:.2f}% ({self.total_planning_attempts - self.planning_failures}/{self.total_planning_attempts})")
        except IOError as e:
            self.get_logger().error(f"Failed to write summary file: {e}")

        grasp_summary_file_path = os.path.join(self.result_path, "grasp_summary.csv")
        try:
            with open(grasp_summary_file_path, 'w') as f:
                f.write("detect_flag,grasp_pose_x,grasp_pose_y,grasp_pose_z,"
                        "grasp_pose_qx,grasp_pose_qy,grasp_pose_qz,grasp_pose_qw,score\n")
                self.get_logger().info(f"final_grasp_succeeded: {self.final_grasp_succeeded}, final_grasp_pose: {self.final_grasp_pose}, final_grasp_score: {self.final_grasp_score}")
                               
                if self.final_grasp_succeeded and self.final_grasp_pose is not None:
                    detect_flag = 1
                    pose_str = ",".join(map(str, self.final_grasp_pose))
                    score = self.final_grasp_score if self.final_grasp_score is not None else ""
                    f.write(f"{detect_flag},{pose_str},{score}\n")
                else:
                    detect_flag = 0
                    f.write(f"{detect_flag},,,,,,,,\n")
            
            self.get_logger().info(f"Grasp summary saved to {grasp_summary_file_path}")
        except IOError as e:
            self.get_logger().error(f"Failed to write grasp summary file: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationCoordinator()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = node.create_rate(5)

    try:

        node.get_logger().info("Now waiting for start signal...")
        while rclpy.ok() and not node.start_exploration_signal:
            time.sleep(0.1) 

        if node.start_exploration_signal:
             node.current_state = node.STATE_AWARENESS_WRIST_SCAN

        while rclpy.ok() and node.current_state != node.STATE_FINISHED:

            state_to_process = node.current_state

            node.get_logger().info(f"--- Processing state callback: {state_to_process} ---", throttle_duration_sec=2.0)

            if state_to_process == node.STATE_AWARENESS_WRIST_SCAN:
                node.execute_awareness_phase(node.STATE_REQUESTING_NBV, "Wrist Scan")

            elif state_to_process == node.STATE_REQUESTING_NBV:
                node.request_nbv_and_move(GetNBV.Request.EXPLORE_FULL, 0.4)

            elif state_to_process == node.STATE_MOVE_TO_SEE:
                with node.detection_data_lock:
                    detection_data_to_execute = node.latest_detection_data
                if detection_data_to_execute:
                    node.execute_move_to_see(detection_data_to_execute)
                else:
                    node.get_logger().error("Entered MOVE_TO_SEE state but no detection data. Returning to exploration.")
                    node.current_state = node.STATE_REQUESTING_NBV

            elif state_to_process == node.STATE_WAITING_FOR_GRASP:
                node.get_logger().info("callback: step1: Waiting for GraspNet result...")
                
                grasp_data_to_check = None
                with node.grasp_data_lock:
                    if node.latest_grasp_data is not None:
                        grasp_data_to_check = node.latest_grasp_data

                elapsed_time = (node.get_clock().now() - node.wait_start_time).nanoseconds / 1e9

                if grasp_data_to_check:
                    trigger_msg = Bool()
                    trigger_msg.data = False
                    node.graspnet_trigger_publisher.publish(trigger_msg)
                    node.get_logger().info("callback: step3: Grasp result received from GraspNet.")

                    if grasp_data_to_check.get('grasps'):
                        node.get_logger().info("callback: Valid grasp received! Transitioning to GRASPING state.")
                        node.current_state = node.STATE_GRASPING
                    else:
                        node.get_logger().warn("callback: GraspNet reported no valid grasps found (result was empty). Returning to exploration.")
                        node.current_state = node.STATE_REQUESTING_NBV

                    node.get_logger().info("callback: step4: logic processed!")

                elif elapsed_time > node.grasp_wait_timeout:
                    trigger_msg = Bool()
                    trigger_msg.data = False
                    node.graspnet_trigger_publisher.publish(trigger_msg)
                    node.get_logger().error(f"callback: Timeout! Waited {elapsed_time:.2f}s for grasp result, but received none. Returning to exploration.")
                    node.current_state = node.STATE_REQUESTING_NBV

            elif state_to_process == node.STATE_GRASPING:
                node.get_logger().info("callback: --- Executing Grasping State ---")
                with node.grasp_data_lock:
                    grasp_data_to_execute = node.latest_grasp_data
                if grasp_data_to_execute:
                    node.execute_grasp(grasp_data_to_execute)
                else:
                    node.get_logger().error("Entered GRASPING state but no grasp data available. Finishing.")
                    node.current_state = node.STATE_FINISHED

            if node.current_state == node.STATE_REQUESTING_NBV:
                detection_found = False
                grasp_found = False

                with node.detection_data_lock:
                    if node.latest_detection_data and node.latest_detection_data[0] == 1.0:
                        detection_found = True

                if not detection_found:
                    with node.grasp_data_lock:
                        if node.latest_grasp_data and node.latest_grasp_data[0] == 1.0:
                            grasp_found = True

                if detection_found:
                    node.get_logger().info("\033[1;32mValid object detection! Transitioning to MOVE_TO_SEE state.\033[0m")
                    node.current_state = node.STATE_MOVE_TO_SEE
                elif grasp_found:
                    node.get_logger().info("\033[1;32mValid grasp detected! Transitioning to GRASPING state.\033[0m")
                    node.current_state = node.STATE_GRASPING

            rate.sleep()

    except (KeyboardInterrupt, RuntimeError) as e:
        node.get_logger().info(f"Execution stopped due to: {e}")
    finally:
        node.get_logger().info("Shutting down...")
        shutdown_trigger = Bool()
        shutdown_trigger.data = False
        node.graspnet_trigger_publisher.publish(shutdown_trigger)

        node.on_shutdown()
        
        node.destroy_node()
        executor.shutdown()
        
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()
