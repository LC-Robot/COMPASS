#include <memory>
#include <thread>
#include <chrono>
#include <mutex>
#include <sstream>
#include <vector>
#include <string> // Added for string manipulation
#include <cstdlib>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <Eigen/Geometry>
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "moveit_planning_service/action/move_to_pose.hpp"
#include "moveit_planning_service/action/move_to_joints.hpp"

#include <yaml-cpp/yaml.h>
#include <fstream>

#include <geometric_shapes/shapes.h>
#include <geometric_shapes/shape_messages.h>
#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>

using MoveToPose = moveit_planning_service::action::MoveToPose;
using GoalHandleMoveToPose = rclcpp_action::ServerGoalHandle<MoveToPose>;
using MoveToJoints = moveit_planning_service::action::MoveToJoints;
using GoalHandleMoveToJoints = rclcpp_action::ServerGoalHandle<MoveToJoints>;

// Helper function to parse pose from YAML
geometry_msgs::msg::Pose parsePoseFromYaml(const YAML::Node& node) {
    geometry_msgs::msg::Pose pose;
    if (node["position"] && node["position"].IsSequence() && node["position"].size() == 3) {
        pose.position.x = node["position"][0].as<double>();
        pose.position.y = node["position"][1].as<double>();
        pose.position.z = node["position"][2].as<double>();
    } else {
        RCLCPP_WARN(rclcpp::get_logger("MotionPlannerServer"), "YAML node missing or invalid 'position' for pose.");
    }

    if (node["orientation"] && node["orientation"].IsSequence() && node["orientation"].size() == 4) {
        // YAML's orientation is usually w, x, y, z. geometry_msgs::msg::Quaternion is x, y, z, w.
        pose.orientation.w = node["orientation"][0].as<double>();
        pose.orientation.x = node["orientation"][1].as<double>();
        pose.orientation.y = node["orientation"][2].as<double>();
        pose.orientation.z = node["orientation"][3].as<double>();
    } else {
        RCLCPP_WARN(rclcpp::get_logger("MotionPlannerServer"), "YAML node missing or invalid 'orientation' for pose.");
        // Default to identity if not found
        pose.orientation.w = 1.0;
        pose.orientation.x = 0.0;
        pose.orientation.y = 0.0;
        pose.orientation.z = 0.0;
    }
    return pose;
}


class MotionPlannerServer : public rclcpp::Node
{
public:
    MotionPlannerServer(const rclcpp::NodeOptions & options) : Node("motion_planner_server", options)
    {
        // Declare the parameter for collision object YAML path
        // this->declare_parameter<std::string>("collision_objects_yaml_path", "");

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&MotionPlannerServer::initialize, this));
    }

private:
    void initialize()
    {
        init_timer_->cancel();
        RCLCPP_INFO(this->get_logger(), "Initializing Motion Planner Server...");

        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(this->shared_from_this(), "panda_arm");
        planning_scene_interface_ = std::make_shared<moveit::planning_interface::PlanningSceneInterface>();
        
        if (move_group_->getPlanningFrame().empty())
        {
            RCLCPP_FATAL(this->get_logger(), "MoveGroupInterface failed to initialize. Shutting down.");
            rclcpp::shutdown();
            return;
        }

        this->pose_action_server_ = rclcpp_action::create_server<MoveToPose>(this, "move_to_pose", std::bind(&MotionPlannerServer::handle_pose_goal, this, std::placeholders::_1, std::placeholders::_2), std::bind(&MotionPlannerServer::handle_pose_cancel, this, std::placeholders::_1), std::bind(&MotionPlannerServer::handle_pose_accepted, this, std::placeholders::_1));
        this->joints_action_server_ = rclcpp_action::create_server<MoveToJoints>(this, "move_to_joints", std::bind(&MotionPlannerServer::handle_joints_goal, this, std::placeholders::_1, std::placeholders::_2), std::bind(&MotionPlannerServer::handle_joints_cancel, this, std::placeholders::_1), std::bind(&MotionPlannerServer::handle_joints_accepted, this, std::placeholders::_1));
        
        // --- 初始化硬编码变换 1: T_link8_camera (从 panda_link8 到 realsense_camera) ---
        T_link8_camera_ = Eigen::Isometry3d::Identity();
        T_link8_camera_.matrix()(0, 0) = 0.707; T_link8_camera_.matrix()(0, 1) = -0.707; T_link8_camera_.matrix()(0, 2) = 0.0;
        T_link8_camera_.matrix()(1, 0) = 0.707; T_link8_camera_.matrix()(1, 1) = 0.707;  T_link8_camera_.matrix()(1, 2) = 0.0;
        T_link8_camera_.matrix()(2, 0) = 0.0;   T_link8_camera_.matrix()(2, 1) = 0.0;    T_link8_camera_.matrix()(2, 2) = 1.0;
        T_link8_camera_.translation() = Eigen::Vector3d(0.035, -0.035, 0.050);
        RCLCPP_INFO(this->get_logger(), "Hardcoded link8->camera transform initialized.");
        std::stringstream ss1;
        ss1 << "T_link8_camera_:\n" << T_link8_camera_.matrix();
        RCLCPP_INFO(this->get_logger(), "%s", ss1.str().c_str());

        // --- 初始化硬编码变换 2: T_camera_world_camera (从 realsense_camera_world 到 realsense_camera) ---
        T_camera_world_to_camera_ = Eigen::Isometry3d::Identity();
        // 使用 .linear() 和 .translation() 更安全
        Eigen::Matrix3d rotation_matrix;
        rotation_matrix << 0.0, 0.0, 1.0,
                           -1.0, 0.0, 0.0,
                           0.0, -1.0, 0.0;
        T_camera_world_to_camera_.linear() = rotation_matrix;
        T_camera_world_to_camera_.translation() = Eigen::Vector3d(0.0, 0.0, 0.0);
        RCLCPP_INFO(this->get_logger(), "Hardcoded camera_world->camera transform initialized.");
        std::stringstream ss2;
        ss2 << "T_camera_world_to_camera_:\n" << T_camera_world_to_camera_.matrix();
        RCLCPP_INFO(this->get_logger(), "%s", ss2.str().c_str());
        
        rclcpp::sleep_for(std::chrono::milliseconds(500));
        this->addCollisionObjectsFromYaml(); // Call the new function

        RCLCPP_INFO(this->get_logger(), "Motion Planner Server has been started and is ready.");
    }

    void addCollisionObjectsFromYaml()
    {
        std::string yaml_path;
        this->get_parameter("collision_objects_yaml_path", yaml_path);
        std::vector<moveit_msgs::msg::CollisionObject> collision_objects;

        if (yaml_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Collision objects YAML path is empty. Adding only the default ground plane.");
            // <--- 修正点 1: 创建一个vector，调用函数，并应用场景 --->
            addGroundPlane(collision_objects);
            planning_scene_interface_->applyCollisionObjects(collision_objects);
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Loading collision objects from YAML file: %s", yaml_path.c_str());

        YAML::Node config;
        try {
            config = YAML::LoadFile(yaml_path);
        } catch (const YAML::BadFile& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load YAML file '%s': %s. Adding only the default ground plane.", yaml_path.c_str(), e.what());
            // <--- 修正点 2: 同样，创建vector，调用函数，并应用场景 --->
            addGroundPlane(collision_objects);
            planning_scene_interface_->applyCollisionObjects(collision_objects);
            return;
        }

        const std::string& planning_frame = move_group_->getPlanningFrame();

        // 总是添加地面
        addGroundPlane(collision_objects); // 这个调用本来就是正确的

        if (config["prims"]) {
            for (YAML::const_iterator it = config["prims"].begin(); it != config["prims"].end(); ++it) {
                std::string object_id = it->first.as<std::string>();
                const YAML::Node& object_node = it->second;

                geometry_msgs::msg::Pose object_pose = parsePoseFromYaml(object_node);

                if (object_id == "target_box") {
                    // ... (这部分代码保持不变) ...
                    moveit_msgs::msg::CollisionObject mesh_object;
                    mesh_object.header.frame_id = planning_frame;
                    mesh_object.id = "banana_mesh";
                    mesh_object.operation = mesh_object.ADD;

                    const char * compass_root_env = std::getenv("COMPASS_ROOT");
                    std::string compass_root = compass_root_env ? compass_root_env : "";
                    std::string mesh_file_path = compass_root.empty()
                        ? "package://moveit_planning_service/../../../../assets/banana.stl"
                        : "file://" + compass_root + "/assets/banana.stl";
                    RCLCPP_INFO(this->get_logger(), "Loading mesh from: %s for object: %s", mesh_file_path.c_str(), object_id.c_str());

                    const auto mesh_shape = std::unique_ptr<shapes::Mesh>(shapes::createMeshFromResource(mesh_file_path));
                    if (!mesh_shape) {
                        RCLCPP_ERROR(this->get_logger(), "Failed to load mesh resource: %s for object: %s", mesh_file_path.c_str(), object_id.c_str());
                        continue;
                    } else {
                        shapes::ShapeMsg shape_variant_msg;
                        shapes::constructMsgFromShape(mesh_shape.get(), shape_variant_msg);
                        shape_msgs::msg::Mesh mesh_msg = boost::get<shape_msgs::msg::Mesh>(shape_variant_msg);
                        
                        mesh_object.meshes.push_back(mesh_msg);
                        mesh_object.mesh_poses.push_back(object_pose);
                        collision_objects.push_back(mesh_object);
                        RCLCPP_INFO(this->get_logger(), "Added mesh object '%s' from YAML at [x:%.2f, y:%.2f, z:%.2f]",
                                    object_id.c_str(), object_pose.position.x, object_pose.position.y, object_pose.position.z);
                    }
                } else if (object_id.rfind("my_custom_box", 0) == 0 || object_id.rfind("obstacle", 0) == 0) {
                    // ... (这部分代码保持不变) ...
                    moveit_msgs::msg::CollisionObject box_object;
                    box_object.header.frame_id = planning_frame;
                    box_object.id = object_id;
                    box_object.operation = box_object.ADD;

                    if (object_node["size"] && object_node["size"].IsSequence() && object_node["size"].size() == 3) {
                        shape_msgs::msg::SolidPrimitive primitive;
                        primitive.type = primitive.BOX;
                        primitive.dimensions.resize(3);
                        primitive.dimensions[primitive.BOX_X] = object_node["size"][0].as<double>();
                        primitive.dimensions[primitive.BOX_Y] = object_node["size"][1].as<double>();
                        primitive.dimensions[primitive.BOX_Z] = object_node["size"][2].as<double>();
                        
                        box_object.primitives.push_back(primitive);
                        box_object.primitive_poses.push_back(object_pose);
                        collision_objects.push_back(box_object);
                        RCLCPP_INFO(this->get_logger(), "Added box object '%s' from YAML at [x:%.2f, y:%.2f, z:%.2f] with size [x:%.2f, y:%.2f, z:%.2f]",
                                    object_id.c_str(), object_pose.position.x, object_pose.position.y, object_pose.position.z,
                                    primitive.dimensions[primitive.BOX_X], primitive.dimensions[primitive.BOX_Y], primitive.dimensions[primitive.BOX_Z]);
                    } else {
                        RCLCPP_WARN(this->get_logger(), "YAML node '%s' missing or invalid 'size' for box object. Skipping.", object_id.c_str());
                    }
                } else {
                    RCLCPP_INFO(this->get_logger(), "Skipping unknown collision object type '%s' from YAML.", object_id.c_str());
                }
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "No 'prims' section found in the collision objects YAML file.");
        }

        if (!collision_objects.empty()) {
            planning_scene_interface_->applyCollisionObjects(collision_objects);
            RCLCPP_INFO(this->get_logger(), "Applied all collision objects to the planning scene.");
        } else {
            RCLCPP_INFO(this->get_logger(), "No collision objects were available to apply.");
        }
    }

    // Helper to add the ground plane (now callable from different places)
    void addGroundPlane(std::vector<moveit_msgs::msg::CollisionObject>& collision_objects_vec)
    {
        moveit_msgs::msg::CollisionObject ground_plane;
        ground_plane.header.frame_id = move_group_->getPlanningFrame();
        ground_plane.id = "ground_plane";
        ground_plane.operation = ground_plane.ADD;
        shape_msgs::msg::SolidPrimitive primitive;
        primitive.type = primitive.BOX;
        primitive.dimensions.resize(3);
        primitive.dimensions[primitive.BOX_X] = 4.0;
        primitive.dimensions[primitive.BOX_Y] = 4.0;
        primitive.dimensions[primitive.BOX_Z] = 0.01;
        geometry_msgs::msg::Pose ground_pose;
        ground_pose.position.x = 0.0;
        ground_pose.position.y = 0.0;
        ground_pose.position.z = -0.005;
        ground_pose.orientation.w = 1.0;
        ground_plane.primitives.push_back(primitive);
        ground_plane.primitive_poses.push_back(ground_pose);
        collision_objects_vec.push_back(ground_plane);
        RCLCPP_INFO(this->get_logger(), "Added default ground plane.");
    }
    
    // Original addCollisionObject is removed, functionality moved to addCollisionObjectsFromYaml and addGroundPlane
    // void addCollisionObject() { /* ... removed ... */ }

    // --- Action回调函数 ---
    rclcpp_action::GoalResponse handle_pose_goal(const rclcpp_action::GoalUUID &, std::shared_ptr<const MoveToPose::Goal>) { return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE; }
    rclcpp_action::CancelResponse handle_pose_cancel(const std::shared_ptr<GoalHandleMoveToPose>) { return rclcpp_action::CancelResponse::ACCEPT; }
    void handle_pose_accepted(const std::shared_ptr<GoalHandleMoveToPose> goal_handle) { std::thread{std::bind(&MotionPlannerServer::execute_pose, this, std::placeholders::_1), goal_handle}.detach(); }
    rclcpp_action::GoalResponse handle_joints_goal(const rclcpp_action::GoalUUID &, std::shared_ptr<const MoveToJoints::Goal>) { return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE; }
    rclcpp_action::CancelResponse handle_joints_cancel(const std::shared_ptr<GoalHandleMoveToJoints>) { return rclcpp_action::CancelResponse::ACCEPT; }
    void handle_joints_accepted(const std::shared_ptr<GoalHandleMoveToJoints> goal_handle) { std::thread{std::bind(&MotionPlannerServer::execute_joints, this, std::placeholders::_1), goal_handle}.detach(); }
    
    // --- 【修改后】的位姿执行函数 ---
    void execute_pose(const std::shared_ptr<GoalHandleMoveToPose> goal_handle)
    {
        std::lock_guard<std::mutex> lock(execution_mutex_);
        const auto goal = goal_handle->get_goal();
        auto result = std::make_shared<MoveToPose::Result>();
        RCLCPP_INFO(this->get_logger(), "Executing goal to move to POSE.");

        // 步骤 1: 提取目标位姿 (T_world_camera_world_goal)
        geometry_msgs::msg::Pose camera_world_target_pose = goal->target_pose.pose;
        Eigen::Isometry3d T_world_camera_world_goal;
        tf2::fromMsg(camera_world_target_pose, T_world_camera_world_goal);

        // 步骤 2: 应用变换链计算最终的 link8 目标
        // 公式: T_world_link8_final = T_world_camera_world_goal * T_camera_world_camera * T_link8_camera.inverse()
        Eigen::Isometry3d T_world_link8_final = T_world_camera_world_goal * T_camera_world_to_camera_ * T_link8_camera_.inverse();

        // 将最终结果转换回 ROS Pose 消息
        geometry_msgs::msg::Pose final_link8_target_pose = tf2::toMsg(T_world_link8_final);
        
        RCLCPP_INFO(this->get_logger(), "Original camera_world target [x: %.3f, y: %.3f, z: %.3f]", 
            camera_world_target_pose.position.x, camera_world_target_pose.position.y, camera_world_target_pose.position.z);
        RCLCPP_INFO(this->get_logger(), "Final transformed link8 target [x: %.3f, y: %.3f, z: %.3f]", 
            final_link8_target_pose.position.x, final_link8_target_pose.position.y, final_link8_target_pose.position.z);

        // 步骤 3: 设置末端执行器和最终目标位姿
        move_group_->setEndEffectorLink("panda_link8");
        bool set_target_ok = move_group_->setPoseTarget(final_link8_target_pose);

        if (!set_target_ok) {
            RCLCPP_ERROR(this->get_logger(), "Failed to set final pose target for link8. It might be unreachable.");
            result->success = false;
            result->message = "Failed to set final pose target. It might be unreachable.";
            goal_handle->abort(result);
            return;
        }
        
        // 步骤 4: 规划和执行
        bool success = (move_group_->move() == moveit::core::MoveItErrorCode::SUCCESS);

        // 步骤 5: 反馈结果
        if (success) {
            RCLCPP_INFO(this->get_logger(), "Execution successful for POSE goal.");
            result->success = true;
            result->message = "Execution successful";
            goal_handle->succeed(result);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Execution failed for POSE goal!");
            result->success = false;
            result->message = "Execution failed";
            goal_handle->abort(result);
        }
    }

    void execute_joints(const std::shared_ptr<GoalHandleMoveToJoints> goal_handle)
    {
        std::lock_guard<std::mutex> lock(execution_mutex_);
        const auto goal = goal_handle->get_goal();
        auto result = std::make_shared<MoveToJoints::Result>();
        RCLCPP_INFO(this->get_logger(), "Executing goal to move to JOINTS.");

        bool set_target_ok = move_group_->setJointValueTarget(goal->target_joints);
        if (!set_target_ok) {
            RCLCPP_ERROR(this->get_logger(), "Failed to set joint value target.");
            result->success = false; result->message = "Failed to set joint value target.";
            goal_handle->abort(result);
            return;
        }

        std::string planner_id = "RRTConnect"; 
        move_group_->setPlannerId(planner_id);
        RCLCPP_INFO(this->get_logger(), "Using planner: %s", planner_id.c_str());
        
        bool success = (move_group_->move() == moveit::core::MoveItErrorCode::SUCCESS);

        if (success) {
            RCLCPP_INFO(this->get_logger(), "Execution successful for JOINTS goal.");
            result->success = true; result->message = "Execution successful";
            goal_handle->succeed(result);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Execution failed for JOINTS goal!");
            result->success = false; result->message = "Execution failed";
            goal_handle->abort(result);
        }
    }
    
    rclcpp::TimerBase::SharedPtr init_timer_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
    rclcpp_action::Server<MoveToPose>::SharedPtr pose_action_server_;
    rclcpp_action::Server<MoveToJoints>::SharedPtr joints_action_server_;
    std::mutex execution_mutex_;

    Eigen::Isometry3d T_link8_camera_;
    Eigen::Isometry3d T_camera_world_to_camera_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto action_server_node = std::make_shared<MotionPlannerServer>(node_options);
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(action_server_node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
