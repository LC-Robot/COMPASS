// ==================================================================
// ======== ROS 2核心和消息头文件 =========
// ==================================================================
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "octomap_msgs/msg/octomap.hpp"
#include "std_msgs/msg/float64.hpp"

// ==================================================================
// ======== 核心依赖库头文件 =========
// ==================================================================
#include <octomap_msgs/conversions.h>
#include <octomap/octomap.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/planning_scene/planning_scene.hpp>

// ==================================================================
// ======== C++ 标准库头文件 =========
// ==================================================================
#include <mutex>
#include <chrono>
#include <random>
#include <Eigen/Geometry>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <limits>

// ==================================================================
// ======== ROS 2 自定义服务头文件 =========
// ==================================================================
#include "nbv_explorer/srv/get_nbv.hpp"
#include "nbv_explorer/srv/get_initial_coverage.hpp"
#include "nbv_explorer/srv/update_weights.hpp"

// --- 新增: TF2 相关头文件 ---
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp" // 用于转换，非常方便

// 使用 using 别名简化代码
using GetNBV = nbv_explorer::srv::GetNBV;
using GetInitialCoverage = nbv_explorer::srv::GetInitialCoverage;
using UpdateWeights = nbv_explorer::srv::UpdateWeights;

using Point = geometry_msgs::msg::Point;
using Pose = geometry_msgs::msg::Pose;
using PoseArray = geometry_msgs::msg::PoseArray;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using Marker = visualization_msgs::msg::Marker;

class StatisticsManager {
public:
    StatisticsManager(rclcpp::Node* parent_node, int _id, int _level, int _scene) 
        : node_(parent_node) 
    {
        const char * experiments_env = std::getenv("COMPASS_EXPERIMENTS_DIR");
        const char * root_env = std::getenv("COMPASS_ROOT");
        std::filesystem::path experiments_root = experiments_env
            ? std::filesystem::path(experiments_env)
            : (root_env ? std::filesystem::path(root_env) / "experiments" : std::filesystem::path("experiments"));
        std::string method_path = (experiments_root / "our_rrt" / ("level" + std::to_string(_level)) / ("scene_" + std::to_string(_scene))).string();
        result_path_ = method_path + "/run_" + std::to_string(_id);
        try { std::filesystem::create_directories(result_path_); }
        catch (const std::filesystem::filesystem_error& e) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to create run directory %s: %s", result_path_.c_str(), e.what());
            return;
        }
        explored_volume_pub_ = node_->create_publisher<std_msgs::msg::Float64>("/exploration/explored_volume", 10);
        path_length_pub_ = node_->create_publisher<std_msgs::msg::Float64>("/exploration/path_length", 10);
        volume_log_.open(result_path_ + "/explored_volume_vs_time.csv", std::ofstream::out | std::ofstream::trunc);
        path_log_.open(result_path_ + "/path_length_vs_time.csv", std::ofstream::out | std::ofstream::trunc);
        manipulability_log_.open(result_path_ + "/manipulability_vs_time.csv", std::ofstream::out | std::ofstream::trunc);
        target_coverage_log_.open(result_path_ + "/target_coverage_vs_time.csv", std::ofstream::out | std::ofstream::trunc);
        volume_log_ << "timestamp,elapsed_time,volume_m3" << std::endl;
        path_log_ << "timestamp,elapsed_time,path_length_m" << std::endl;
        manipulability_log_ << "timestamp,elapsed_time,manipulability" << std::endl;
        target_coverage_log_ << "timestamp,elapsed_time,coverage_ratio" << std::endl;
        start_time_ = node_->get_clock()->now();
    }
    ~StatisticsManager() = default;
    void updateExploredVolume(const octomap::OcTree* octree) {
        if (!octree || !volume_log_.is_open()) return;
        double voxel_volume = pow(octree->getResolution(), 3);
        unsigned int known_voxels = 0;
        for (auto it = octree->begin_leafs(), end = octree->end_leafs(); it != end; ++it) {
            if (it->getLogOdds() != 0.0) known_voxels++;
        }
        double current_volume = known_voxels * voxel_volume;
        rclcpp::Time now = node_->get_clock()->now();
        double elapsed_time = (now - start_time_).seconds();
        std_msgs::msg::Float64 msg;
        msg.data = current_volume;
        explored_volume_pub_->publish(msg);
        volume_log_ << now.nanoseconds() << "," << elapsed_time << "," << current_volume << std::endl;
    }
    void updatePathLength(const Point& new_position) {
        if (!path_log_.is_open()) return;
        if (first_pose_) {
            last_position_ = new_position;
            first_pose_ = false;
        } else {
            double dx = new_position.x - last_position_.x;
            double dy = new_position.y - last_position_.y;
            double dz = new_position.z - last_position_.z;
            total_path_length_ += sqrt(dx*dx + dy*dy + dz*dz);
            last_position_ = new_position;
        }
        rclcpp::Time now = node_->get_clock()->now();
        double elapsed_time = (now - start_time_).seconds();
        std_msgs::msg::Float64 msg;
        msg.data = total_path_length_;
        path_length_pub_->publish(msg);
        path_log_ << now.nanoseconds() << "," << elapsed_time << "," << total_path_length_ << std::endl;
    }
    void updateManipulability(const moveit::core::RobotState& current_state, const moveit::core::JointModelGroup* joint_model_group) {
        if (!manipulability_log_.is_open() || !joint_model_group) return;
        Eigen::MatrixXd jacobian = current_state.getJacobian(joint_model_group);
        Eigen::MatrixXd jj_transpose = jacobian * jacobian.transpose();
        double determinant = jj_transpose.determinant();
        double manipulability = (determinant > 1e-9) ? sqrt(determinant) : 0.0;
        rclcpp::Time now = node_->get_clock()->now();
        double elapsed_time = (now - start_time_).seconds();
        manipulability_log_ << now.nanoseconds() << "," << elapsed_time << "," << manipulability << std::endl;
    }
    void checkTargetVisibility(const octomap::OcTree* octree, const Pose& camera_pose, const Point& target_point) {
        if (!octree || target_found_ || !path_log_.is_open()) return;
        octomap::point3d origin(camera_pose.position.x, camera_pose.position.y, camera_pose.position.z);
        octomap::point3d target(target_point.x, target_point.y, target_point.z);
        double search_radius = 0.05;
        const double resolution = octree->getResolution();
        bool target_neighborhood_observed = false;
        for (double dx = -search_radius; dx <= search_radius; dx += resolution) {
            for (double dy = -search_radius; dy <= search_radius; dy += resolution) {
                for (double dz = -search_radius; dz <= search_radius; dz += resolution) {
                    octomap::point3d p = target + octomap::point3d(dx, dy, dz);
                    if (target.distance(p) > search_radius) continue;
                    if (octree->search(p) != nullptr) {
                        target_neighborhood_observed = true;
                        break;
                    }
                }
                if (target_neighborhood_observed) break;
            }
            if (target_neighborhood_observed) break;
        }
        if (target_neighborhood_observed) {
            rclcpp::Time now = node_->get_clock()->now();
            double elapsed_time = (now - start_time_).seconds();
            RCLCPP_INFO(node_->get_logger(), "\033[1;33mTARGET FOUND! Time: %.2f s, Path Length: %.2f m\033[0m", elapsed_time, total_path_length_);
            std::ofstream target_log(result_path_ + "/target_found_log.csv", std::ofstream::out | std::ofstream::trunc);
            target_log << "timestamp,elapsed_time,path_length_m" << std::endl;
            target_log << now.nanoseconds() << "," << elapsed_time << "," << total_path_length_ << std::endl;
            target_log.close();
            target_found_ = true;
        }
    }
    void updateTargetCoverage(const octomap::OcTree* octree, const Point& target_point, double coverage_radius) {
        if (!octree || target_found_ || !target_coverage_log_.is_open()) return;
        int total_voxels = 0, known_voxels = 0;
        const double resolution = octree->getResolution();
        octomap::point3d target(target_point.x, target_point.y, target_point.z);
        for (double x = target.x() - coverage_radius; x <= target.x() + coverage_radius; x += resolution) {
            for (double y = target.y() - coverage_radius; y <= target.y() + coverage_radius; y += resolution) {
                for (double z = target.z() - coverage_radius; z <= target.z() + coverage_radius; z += resolution) {
                    octomap::point3d p(x, y, z);
                    if (target.distance(p) > coverage_radius) continue;
                    if(z < 0) continue;
                    total_voxels++;
                    if (octree->search(p) != nullptr) {
                        known_voxels++;
                    }
                }
            }
        }
        double coverage_ratio = (total_voxels == 0) ? 1.0 : static_cast<double>(known_voxels) / total_voxels;
        rclcpp::Time now = node_->get_clock()->now();
        double elapsed_time = (now - start_time_).seconds();
        target_coverage_log_ << now.nanoseconds() << "," << elapsed_time << "," << coverage_ratio << std::endl;
    }
private:
    rclcpp::Node* node_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr explored_volume_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr path_length_pub_;
    rclcpp::Time start_time_;
    bool first_pose_ = true;
    Point last_position_;
    double total_path_length_ = 0.0;
    bool target_found_ = false;
    std::ofstream volume_log_, path_log_, manipulability_log_, target_coverage_log_;
    std::string result_path_;
};

struct RRTNode {
    int id;
    Pose pose;
    double gain = 0.0;
    double weighted_gain = 0.0;
    int parent_id = -1;
    double cost_from_root = 0.0;
    double utility = 0.0;
};

class RRTExplorerServer : public rclcpp::Node {
public:
    RRTExplorerServer() : 
        Node("rrt_explorer_node"), 
        random_generator_(std::random_device{}())
    {
        // --- 1. 声明并加载参数 ---
        RCLCPP_INFO(this->get_logger(), "Declaring and loading parameters...");

        // General parameters
        this->declare_parameter<std::string>("world_frame", "panda_link0");
        this->declare_parameter<std::string>("planning_group", "panda_arm");
        this->declare_parameter<std::string>("camera_link_name", "realsense_camera_world");
        this->declare_parameter<double>("frontier_cluster_distance", 0.5);
        this->declare_parameter<double>("viewpoint_generation_radius", 0.8);
        this->declare_parameter<int>("min_cluster_size", 10);
        this->declare_parameter<double>("gain_max_range", 1.0);
        this->declare_parameter<double>("frontier_max_height", 1.2);
        this->declare_parameter<bool>("use_static_robot_box", true);

        // RRT parameters (note the '.' for namespacing)
        this->declare_parameter<int>("rrt.max_iterations", 40);
        this->declare_parameter<double>("rrt.extension_range", 0.3);

        // Bounding box parameters
        this->declare_parameter<double>("bounds.min_x", -1.0);
        this->declare_parameter<double>("bounds.max_x", 1.0);
        this->declare_parameter<double>("bounds.min_y", -1.0);
        this->declare_parameter<double>("bounds.max_y", 1.0);
        this->declare_parameter<double>("bounds.min_z", 0.15);
        this->declare_parameter<double>("bounds.max_z", 1.0);

        // Target finding parameters
        this->declare_parameter<double>("target.coverage_radius", 0.5);
        this->declare_parameter<double>("target.x", 100.0);
        this->declare_parameter<double>("target.y", 100.0);
        this->declare_parameter<double>("target.z", 100.0);

        // Guidance parameters
        this->declare_parameter<double>("guidance.target_bias_probability", 0.5);
        this->declare_parameter<double>("guidance.target_bonus_weight", 2.0);

        // Experiment parameters
        this->declare_parameter<int>("run_id", 0);
        this->declare_parameter<int>("level", 0);
        this->declare_parameter<int>("scene", 0);

        // --- Get all parameters ---

        // General parameters
        this->get_parameter("world_frame", world_frame_);
        this->get_parameter("planning_group", planning_group_);
        this->get_parameter("camera_link_name", camera_link_name_);
        this->get_parameter("frontier_cluster_distance", cluster_distance_threshold_);
        this->get_parameter("viewpoint_generation_radius", generation_radius_);
        this->get_parameter("min_cluster_size", min_cluster_size_);
        this->get_parameter("gain_max_range", gain_max_range_);
        this->get_parameter("frontier_max_height", frontier_max_height_);
        this->get_parameter("use_static_robot_box", use_static_robot_box_);

        // RRT parameters
        this->get_parameter("rrt.max_iterations", rrt_max_iterations_);
        this->get_parameter("rrt.extension_range", rrt_extension_range_);

        // Bounding box parameters
        this->get_parameter("bounds.min_x", min_bound_x_);
        this->get_parameter("bounds.max_x", max_bound_x_);
        this->get_parameter("bounds.min_y", min_bound_y_);
        this->get_parameter("bounds.max_y", max_bound_y_);
        this->get_parameter("bounds.min_z", min_bound_z_);
        this->get_parameter("bounds.max_z", max_bound_z_);

        // Target finding parameters
        this->get_parameter("target.coverage_radius", target_coverage_radius_);
        this->get_parameter("target.x", target_to_find_.x);
        this->get_parameter("target.y", target_to_find_.y);
        this->get_parameter("target.z", target_to_find_.z);

        // Guidance parameters
        this->get_parameter("guidance.target_bias_probability", target_bias_probability_);
        this->get_parameter("guidance.target_bonus_weight", target_bonus_weight_);

        // Experiment parameters (local variables for constructor)
        int run_id, level, scene;
        this->get_parameter("run_id", run_id);
        this->get_parameter("level", level);
        this->get_parameter("scene", scene);

        RCLCPP_INFO(this->get_logger(), "All parameters loaded successfully.");

        // --- 2. 【修改】在构造函数中初始化硬编码的变换矩阵 ---
        // 变换 1: T_link8_camera (从 panda_link8 到 realsense_camera)  Tlc
        T_link8_camera_ = Eigen::Isometry3d::Identity();
        T_link8_camera_.matrix()(0, 0) = 0.707; T_link8_camera_.matrix()(0, 1) = -0.707; T_link8_camera_.matrix()(0, 2) = 0.0;
        T_link8_camera_.matrix()(1, 0) = 0.707; T_link8_camera_.matrix()(1, 1) = 0.707;  T_link8_camera_.matrix()(1, 2) = 0.0;
        T_link8_camera_.matrix()(2, 0) = 0.0;   T_link8_camera_.matrix()(2, 1) = 0.0;    T_link8_camera_.matrix()(2, 2) = 1.0;
        T_link8_camera_.translation() = Eigen::Vector3d(0.035, -0.035, 0.050);
        RCLCPP_INFO(this->get_logger(), "Hardcoded link8->camera transform initialized.");
        
        // 变换 2: T_camera_world_to_camera (从 realsense_camera_world 到 realsense_camera)
        T_camera_world_to_camera_ = Eigen::Isometry3d::Identity();
        Eigen::Matrix3d rotation_matrix;
        rotation_matrix << 0.0, 0.0, 1.0,
                           -1.0, 0.0, 0.0,
                           0.0, -1.0, 0.0;
        T_camera_world_to_camera_.linear() = rotation_matrix;
        T_camera_world_to_camera_.translation() = Eigen::Vector3d(0.0, 0.0, 0.0);
        RCLCPP_INFO(this->get_logger(), "Hardcoded camera_world->camera transform initialized.");

        // --- 3. 初始化ROS接口 ---
        octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
            "/octomap_full", 10, std::bind(&RRTExplorerServer::octomapCallback, this, std::placeholders::_1));

        get_nbv_service_ = this->create_service<GetNBV>(
            "get_next_best_viewpoint", std::bind(&RRTExplorerServer::getNBVCallback, this, std::placeholders::_1, std::placeholders::_2));
        
        get_coverage_service_ = this->create_service<GetInitialCoverage>(
            "get_coverage_ratio", std::bind(&RRTExplorerServer::getCoverageCallback, this, std::placeholders::_1, std::placeholders::_2));
            
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&RRTExplorerServer::jointStateCallback, this, std::placeholders::_1));

        target_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/potential_targets", 10, std::bind(&RRTExplorerServer::targetCallback, this, std::placeholders::_1));

        update_weights_service_ = this->create_service<UpdateWeights>(
            "update_exploration_weights", std::bind(&RRTExplorerServer::updateWeightsCallback, this, std::placeholders::_1, std::placeholders::_2));

        // 为"latch"行为的话题设置QoS
        rclcpp::QoS latched_qos(1);
        latched_qos.transient_local();

        rrt_tree_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_tree_vis", 10);
        best_viewpoint_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/best_viewpoint", latched_qos);
        frontier_points_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/frontier_points", latched_qos);
        frontier_clusters_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/frontier_clusters", latched_qos);

        // 初始化TF2 Buffer和Listener
        // this->get_clock() 是获取与节点关联的时钟的正确方式
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // --- 4. 初始化统计管理器 ---
        stats_manager_ = std::make_shared<StatisticsManager>(this, run_id, level, scene);
        
        RCLCPP_INFO(this->get_logger(), "RRT Explorer Server initialized. Ready to provide NBVs.");
    }

    bool init() {
        RCLCPP_INFO(this->get_logger(), "Initializing MoveIt components...");

        // --- 2. 初始化MoveIt!组件 (现在在这里进行) ---
        // RobotModelLoader需要一个节点实例, 现在可以安全地调用 shared_from_this()
        robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(this->shared_from_this(), "robot_description");
        robot_model_ = robot_model_loader_->getModel();
        planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_model_);
        joint_model_group_ = robot_model_->getJointModelGroup(planning_group_);
        if (!joint_model_group_) {
            RCLCPP_FATAL(this->get_logger(), "Planning group '%s' not found.", planning_group_.c_str());
            return false; // 初始化失败
        }

        if (use_static_robot_box_) {
            rclcpp::sleep_for(std::chrono::milliseconds(500));
            calculateInitialPoseBoundingBox();
        }
        
        RCLCPP_INFO(this->get_logger(), "RRT Explorer Server initialized. Ready to provide NBVs.");
        return true; // 初始化成功
    }

private:
    // --- 服务回调：RRT主流程 ---
    void getNBVCallback(const std::shared_ptr<GetNBV::Request> req, std::shared_ptr<GetNBV::Response> res) {
        if (!map_received_) {
            res->success = false;
            res->message = "No OctoMap received.";
            RCLCPP_ERROR(this->get_logger(), "Cannot provide NBV: %s", res->message.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "--- Received RRT-based NBV Request. ---");

        // 1. 获取当前 camera_link_name_ (即 realsense_camera_world) 的位姿作为RRT的根节点
        geometry_msgs::msg::PoseStamped current_pose_stamped;
        try {
            // 确保参数 camera_link_name_ 被设置为 "realsense_camera_world"
            geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
                world_frame_,
                camera_link_name_,
                tf2::TimePointZero,
                std::chrono::seconds(1)
            );
            current_pose_stamped.header.stamp = this->get_clock()->now();
            current_pose_stamped.header.frame_id = world_frame_;
            current_pose_stamped.pose.position.x = transform_stamped.transform.translation.x;
            current_pose_stamped.pose.position.y = transform_stamped.transform.translation.y;
            current_pose_stamped.pose.position.z = transform_stamped.transform.translation.z;
            current_pose_stamped.pose.orientation = transform_stamped.transform.rotation;
        } catch (const tf2::TransformException &ex) {
            res->success = false;
            res->message = "Could not get transform from '" + world_frame_ + "' to '" + 
                        camera_link_name_ + "': " + std::string(ex.what());
            RCLCPP_ERROR(this->get_logger(), "%s", res->message.c_str());
            return;
        }
        
        // 2. 清空并初始化RRT树
        rrt_tree_.clear();
        RRTNode root_node;
        root_node.id = 0;
        root_node.pose = current_pose_stamped.pose; // RRT的节点现在存储的是 camera_world 的位姿
        root_node.parent_id = -1;
        rrt_tree_.push_back(root_node);

        // 3. 寻找前沿用于偏向性采样
        auto clusters = findAndClusterFrontiers();
        if (clusters.empty()) {
            res->success = false; 
            res->message = "No frontiers found.";
            RCLCPP_WARN(this->get_logger(), "NBV selection failed: %s", res->message.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Identified %zu frontier clusters for biased sampling.", clusters.size());

        // 4. 执行RRT生长循环
        for (int i = 0; i < rrt_max_iterations_; ++i) {
            RCLCPP_INFO(this->get_logger(), "RRT iteration %d/%d", (i+1), rrt_max_iterations_);

            geometry_msgs::msg::Pose sampled_pose = samplePose(clusters);
            int nearest_node_id = findNearestNode(sampled_pose);
            const RRTNode& parent_node = rrt_tree_[nearest_node_id];
            // new_pose 是 camera_world 的候选目标位姿
            geometry_msgs::msg::Pose new_pose = steer(parent_node.pose, sampled_pose);

            // 区域和高度限制检查
            if ((req->mode == req->EXPLORE_AROUND || req->mode == req->EXPLORE_WRIST) && 
                (std::hypot(new_pose.position.x, new_pose.position.y, new_pose.position.z) > req->max_distance)) {
                continue;
            }
            if (new_pose.position.z < 0.1) {
                continue;
            }

            // --- 将 camera_world 的目标位姿转换为 link8 的IK目标 ---
            Eigen::Isometry3d T_world_camera_world_goal;
            tf2::fromMsg(new_pose, T_world_camera_world_goal);
            
            // 公式: T_world_link8_final = T_world_camera_world_goal * T_camera_world_camera * T_link8_camera.inverse()
            Eigen::Isometry3d T_world_link8_final = T_world_camera_world_goal * T_camera_world_to_camera_ * T_link8_camera_.inverse();

            geometry_msgs::msg::Pose ik_target_pose_for_link8 = tf2::toMsg(T_world_link8_final);

            // --- 使用计算出的正确目标进行IK求解 ---
            std::string moveit_ik_frame = "panda_link8";
            moveit::core::RobotState new_state = planning_scene_->getCurrentState();

            if (new_state.setFromIK(joint_model_group_, ik_target_pose_for_link8, moveit_ik_frame, 0.05)) {
                if (req->mode == req->EXPLORE_WRIST) {
                    moveit::core::RobotState parent_state = planning_scene_->getCurrentState();
                    
                    // 为父节点计算IK，同样应用完整的变换链
                    Eigen::Isometry3d T_world_camera_world_parent;
                    tf2::fromMsg(parent_node.pose, T_world_camera_world_parent);
                    Eigen::Isometry3d T_world_link8_final_parent = T_world_camera_world_parent * T_camera_world_to_camera_ * T_link8_camera_.inverse();
                    geometry_msgs::msg::Pose ik_target_parent = tf2::toMsg(T_world_link8_final_parent);
                    
                    if(parent_state.setFromIK(joint_model_group_, ik_target_parent, moveit_ik_frame, 0.05)){
                        double base_joints_motion = 0.0;
                        for (int j = 0; j < 4; ++j) {
                            base_joints_motion += std::abs(new_state.getVariablePosition(j) - parent_state.getVariablePosition(j));
                        }
                        if (base_joints_motion > 0.2) {
                            continue; 
                        }
                    }
                }

                if (!planning_scene_->isStateColliding(new_state, joint_model_group_->getName())) {
                    RRTNode new_node;
                    new_node.id = rrt_tree_.size();
                    new_node.pose = new_pose; // 存储的是 camera_world 的位姿
                    new_node.parent_id = nearest_node_id;
                    
                    // evaluateSingleViewpoint 现在需要接收 camera_world 的位姿
                    auto gains = evaluateSingleViewpoint(new_node.pose);
                    new_node.gain = gains.first;
                    new_node.weighted_gain = gains.second;

                    const RRTNode& parent_node_ref = rrt_tree_[nearest_node_id];
                    moveit::core::RobotState parent_state = planning_scene_->getCurrentState();
                    
                    // 再次为父节点计算IK以得到准确的关节状态来计算成本
                    Eigen::Isometry3d T_world_camera_world_parent;
                    tf2::fromMsg(parent_node_ref.pose, T_world_camera_world_parent);
                    Eigen::Isometry3d T_world_link8_final_parent = T_world_camera_world_parent * T_camera_world_to_camera_ * T_link8_camera_.inverse();
                    geometry_msgs::msg::Pose ik_target_parent = tf2::toMsg(T_world_link8_final_parent);

                    if (parent_state.setFromIK(joint_model_group_, ik_target_parent, moveit_ik_frame, 0.05)) {
                        new_node.cost_from_root = parent_node_ref.cost_from_root + parent_state.distance(new_state, joint_model_group_);
                    } else {
                        // 如果父节点IK失败，成本无法计算，跳过此节点
                        RCLCPP_WARN(this->get_logger(), "IK for parent node %d failed, cannot compute cost. Skipping.", parent_node_ref.id);
                        continue;
                    }
                    
                    rrt_tree_.push_back(new_node);
                }
            }
        }
        RCLCPP_INFO(this->get_logger(), "RRT growth finished with %zu nodes.", rrt_tree_.size());
        
        publishRRTTree();

        RRTNode best_node = selectBestNode();

        if (best_node.utility <= 0.0) {
            res->success = false; 
            res->message = "No viewpoint with positive utility found in RRT.";
            RCLCPP_WARN(this->get_logger(), "NBV selection failed: %s", res->message.c_str());
            return;
        }

        res->nbv.header.frame_id = world_frame_;
        res->nbv.header.stamp = this->get_clock()->now();
        res->nbv.pose = best_node.pose; // best_node.pose 现在是 camera_world 的位姿
        res->success = true;
        res->message = "Successfully found a Next-Best-Viewpoint using RRT.";
        
        best_viewpoint_pub_->publish(res->nbv);
        
        RCLCPP_INFO(this->get_logger(), "\033[1;32mSuccessfully provided a RRT-based NBV.\033[0m");
    }

    // --- 订阅者回调 ---
    // 回调签名已改变
    void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg) {
        octomap::AbstractOcTree* tree = octomap_msgs::msgToMap(*msg);
        if (tree) {
            // <<< 核心修复: 使用互斥锁保护地图的写入操作 >>>
            std::lock_guard<std::mutex> lock(octomap_mutex_);
            
            octree_.reset(dynamic_cast<octomap::OcTree*>(tree));
            // planning_scene_->processOctomapMsg(*msg);

            if(!map_received_) RCLCPP_INFO(this->get_logger(), "First OctoMap received!");
            map_received_ = true;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to deserialize OctoMap message.");
        }

        // <<< 新增: 更新统计数据 >>>
        if (stats_manager_) {
            stats_manager_->updateExploredVolume(octree_.get());
        }
    }

    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        // 当收到新的关节状态时，用它来更新我们内部planning_scene中的当前机器人状态
        
        // <<< 核心修复: 加锁以保护planning_scene_ >>>
        std::lock_guard<std::mutex> lock(octomap_mutex_); // 复用同一个锁来保护整个planning scene

        if (planning_scene_) {
            // 获取当前的机器人状态对象
            moveit::core::RobotState& current_state = planning_scene_->getCurrentStateNonConst();
            // 使用收到的消息来设置关节位置
            current_state.setVariablePositions(msg->name, msg->position);

            // <<< 新增: 更新统计数据 >>>
            if (stats_manager_ && current_state.knowsFrameTransform(camera_link_name_)) {
                // a. 获取当前相机位姿
                const Eigen::Isometry3d& transform = current_state.getGlobalLinkTransform(camera_link_name_);
                Pose current_camera_pose = tf2::toMsg(transform);
                
                // b. 更新路径长度
                stats_manager_->updatePathLength(current_camera_pose.position);

                // c. 检查目标可见性
                stats_manager_->checkTargetVisibility(octree_.get(), current_camera_pose, target_to_find_);
            
                stats_manager_->updateManipulability(current_state, joint_model_group_);
            
                stats_manager_->updateTargetCoverage(octree_.get(), target_to_find_, target_coverage_radius_);
            }
        }
    }
    
    void targetCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        // 使用互斥锁来保证对目标列表的写入是线程安全的
        std::lock_guard<std::mutex> lock(target_mutex_);
        
        detected_targets_.clear(); // 先清空旧的目标
        for(const auto& pose : msg->poses) {
            detected_targets_.push_back(pose.position); // 我们只关心位置
        }

        last_target_received_time_ = this->get_clock()->now();

        if (!detected_targets_.empty()) {
            RCLCPP_INFO(this->get_logger(), "\033[1;35mReceived %zu heuristic targets. Exploration is now target-driven.\033[0m", 
                    detected_targets_.size());
        }
    }
    
    // 其他服务回调 (签名已改变)
    void getCoverageCallback(const std::shared_ptr<GetInitialCoverage::Request> req,
                             std::shared_ptr<GetInitialCoverage::Response> res)
    {
        std::lock_guard<std::mutex> lock(octomap_mutex_);
        if (!octree_) {
            res->coverage_ratio = 0.0;
            return;
        }

        if (!use_static_robot_box_) {
            res->coverage_ratio = 1.0; // 如果功能禁用，则假设完成
            return;
        }

        int total_voxels = 0;
        int known_voxels = 0;
        const double resolution = octree_->getResolution();

        for (double x = robot_min_x_; x <= robot_max_x_; x += resolution) {
            for (double y = robot_min_y_; y <= robot_max_y_; y += resolution) {
                for (double z = robot_min_z_; z <= robot_max_z_; z += resolution) {
                    total_voxels++;
                    if (octree_->search(octomap::point3d(x, y, z)) != nullptr) {
                        known_voxels++;
                    }
                }
            }
        }
        
        if (total_voxels == 0) {
            res->coverage_ratio = 1.0;
        } else {
            res->coverage_ratio = static_cast<double>(known_voxels) / total_voxels;
        }

        RCLCPP_INFO(this->get_logger(), "Peri-personal space coverage check: %.1f%% (%d/%d known)",
                res->coverage_ratio * 100.0, known_voxels, total_voxels);
    }
    
    void updateWeightsCallback(const std::shared_ptr<UpdateWeights::Request> req,
                               std::shared_ptr<UpdateWeights::Response> res)
    {
        std::lock_guard<std::mutex> lock(octomap_mutex_); // 复用同一个锁来保护所有地图
        
        if(octree_ && (!weight_map_)){ 
            weight_map_ = std::make_shared<octomap::OcTree>(octree_->getResolution());
        }

        // 清空旧的权重地图
        weight_map_->clear();

        pcl::PointCloud<pcl::PointXYZI>::Ptr weighted_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(req->weighted_points, *weighted_cloud);

        for (const auto& point : *weighted_cloud) {
            // 使用 updateNode 来设置体素的值
            // 我们将权重存储在LogOdds值中。LogOdds是float类型。
            // 注意：OctoMap内部的值是对数赔率，但我们可以直接存入任何浮点数。
            // 确保权重是非负的。
            if (point.intensity >= 0) {
                weight_map_->updateNode(point.x, point.y, point.z, static_cast<float>(point.intensity));
            }
        }

        RCLCPP_INFO(this->get_logger(), "Updated weight map with %zu points.", weighted_cloud->size());
        res->success = true;
    }

    // --- RRT 核心辅助函数 (大部分内部逻辑无变化) ---
    Pose samplePose(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters) { 
        Pose p;

        // --- 步骤A: 检查是否有可用的引导目标 ---
        std::vector<Point> current_targets;
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            // 检查目标列表是否为空，并且信息是否在10秒内更新过 (避免使用过时的引导)
            if (!detected_targets_.empty() )
            //   && (ros::Time::now() - last_target_received_time_).toSec() < 10.0) 
            {
                current_targets = detected_targets_;
            }
        }

        // --- 步骤B: 根据有无目标，决定采样策略 ---
        // 条件: 有引导目标, 并且一个随机数小于我们设定的偏向概率
        if (!current_targets.empty() && uniform_dist_(random_generator_) < target_bias_probability_) {
            // --- 策略1: 目标导向采样 ---
            RCLCPP_INFO_ONCE(this->get_logger(), "Sampling is now biased towards the heuristic target.");

            // a. 随机选择一个检测到的目标作为“朝向”
            int target_idx = uniform_dist_(random_generator_) * current_targets.size();
            const auto& target_point_msg = current_targets[target_idx];
            octomap::point3d target_point(target_point_msg.x, target_point_msg.y, target_point_msg.z);

            // b. 在机器人基座为中心的视图球上采样一个“位置” (与之前相同)
            octomap::point3d robot_base_center(0.0, 0.0, 0.0);
            double phi = acos(1 - 2 * uniform_dist_(random_generator_));
            double theta = 2 * M_PI * uniform_dist_(random_generator_);
            p.position.x = robot_base_center.x() + generation_radius_ * sin(phi) * cos(theta);
            p.position.y = robot_base_center.y() + generation_radius_ * sin(phi) * sin(theta);
            p.position.z = robot_base_center.z() + generation_radius_ * cos(phi);

            // c. 计算姿态，使其指向这个选定的引导目标
            Eigen::Vector3d view_direction(target_point.x() - p.position.x, target_point.y() - p.position.y, target_point.z() - p.position.z);
            if (view_direction.norm() > 1e-6) view_direction.normalize();
            Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitX(), view_direction);
            p.orientation.x = q.x(); p.orientation.y = q.y(); p.orientation.z = q.z(); p.orientation.w = q.w();

        } else {
            // --- 策略2: 前沿导向/纯随机采样 (与之前完全相同) ---
            RCLCPP_INFO_ONCE(this->get_logger(), "Sampling is based on frontiers or random.");

            if (uniform_dist_(random_generator_) > 0.25 && !clusters.empty()) {
                int cluster_idx = uniform_dist_(random_generator_) * clusters.size();
                const auto& cluster = clusters[cluster_idx];
                int point_idx = uniform_dist_(random_generator_) * cluster->size();
                octomap::point3d target_point((*cluster)[point_idx].x, (*cluster)[point_idx].y, (*cluster)[point_idx].z);

                octomap::point3d robot_base_center(0.0, 0.0, 0.0);
                double phi = acos(1 - 2 * uniform_dist_(random_generator_));
                double theta = 2 * M_PI * uniform_dist_(random_generator_);
                p.position.x = robot_base_center.x() + generation_radius_ * sin(phi) * cos(theta);
                p.position.y = robot_base_center.y() + generation_radius_ * sin(phi) * sin(theta);
                p.position.z = robot_base_center.z() + generation_radius_ * cos(phi);

                Eigen::Vector3d view_direction(target_point.x() - p.position.x, target_point.y() - p.position.y, target_point.z() - p.position.z);
                view_direction.normalize();
                Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitX(), view_direction);
                p.orientation.x = q.x(); p.orientation.y = q.y(); p.orientation.z = q.z(); p.orientation.w = q.w();
            } else {
                p.position.x = uniform_dist_(random_generator_) * (max_bound_x_ - min_bound_x_) + min_bound_x_;
                p.position.y = uniform_dist_(random_generator_) * (max_bound_y_ - min_bound_y_) + min_bound_y_;
                p.position.z = uniform_dist_(random_generator_) * (max_bound_z_ - min_bound_z_) + min_bound_z_;
                Eigen::Quaterniond q_rand = Eigen::AngleAxisd(uniform_dist_(random_generator_) * 2 * M_PI, Eigen::Vector3d::UnitZ()) *
                                        Eigen::AngleAxisd(uniform_dist_(random_generator_) * M_PI, Eigen::Vector3d::UnitY());
                p.orientation.x = q_rand.x(); p.orientation.y = q_rand.y(); p.orientation.z = q_rand.z(); p.orientation.w = q_rand.w();
            }
        }
        return p;
    }

    int findNearestNode(const Pose& pose) { 
        int best_id = -1;
        double min_dist_sq = std::numeric_limits<double>::max();
        for (const auto& node : rrt_tree_) {
            double dist_sq = pow(node.pose.position.x - pose.position.x, 2) +
                             pow(node.pose.position.y - pose.position.y, 2) +
                             pow(node.pose.position.z - pose.position.z, 2);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_id = node.id;
            }
        }
        return best_id;
    }

    Pose steer(const Pose& from, const Pose& to) { 
        Pose result = from; // 【已修正】声明在函数开头
        double dist = sqrt(pow(to.position.x - from.position.x, 2) +
                           pow(to.position.y - from.position.y, 2) +
                           pow(to.position.z - from.position.z, 2));
        if (dist < rrt_extension_range_) {
            return to;
        }
        result.position.x = from.position.x + (to.position.x - from.position.x) / dist * rrt_extension_range_;
        result.position.y = from.position.y + (to.position.y - from.position.y) / dist * rrt_extension_range_;
        result.position.z = from.position.z + (to.position.z - from.position.z) / dist * rrt_extension_range_;
        Eigen::Quaterniond q_from(from.orientation.w, from.orientation.x, from.orientation.y, from.orientation.z);
        Eigen::Quaterniond q_to(to.orientation.w, to.orientation.x, to.orientation.y, to.orientation.z);
        Eigen::Quaterniond q_res = q_from.slerp(rrt_extension_range_ / dist, q_to);
        result.orientation.x = q_res.x(); result.orientation.y = q_res.y(); result.orientation.z = q_res.z(); result.orientation.w = q_res.w();
        return result;
    }

    std::pair<double, double> evaluateSingleViewpoint(const geometry_msgs::msg::Pose& camera_world_pose) {
        // =====================================================================================
        // ================= vvv 【修改后】的位姿变换逻辑 vvv ======================================
        // =====================================================================================
        
        // 步骤 1: 将输入的 camera_world_pose 转换为实际的光学中心 camera_pose
        // 这是必需的，因为所有的射线投射（ray casting）都必须从相机光学中心发出。
        Eigen::Isometry3d T_world_camera_world;
        tf2::fromMsg(camera_world_pose, T_world_camera_world);

        // 应用正向变换: T_world_camera = T_world_camera_world * T_camera_world_camera
        Eigen::Isometry3d T_world_camera = T_world_camera_world * T_camera_world_to_camera_;

        // 将计算出的光学中心位姿转换回 ROS Pose 消息，以供后续代码使用
        geometry_msgs::msg::Pose pose = tf2::toMsg(T_world_camera);
        
        // =====================================================================================
        // ================= ^^^ 【修改后】的位姿变换逻辑 ^^^ ======================================
        // =====================================================================================

        // 步骤 2: 使用转换后的光学中心位姿 'pose' 执行信息增益计算
        std::lock_guard<std::mutex> lock(octomap_mutex_);
        if (!octree_) {
            return {0.0, 0.0};
        }
        
        double geometric_gain = 0.0;
        double weighted_gain = 0.0;
        
        // 从转换后的 'pose' 中提取原点和方向
        octomap::point3d origin(pose.position.x, pose.position.y, pose.position.z);
        Eigen::Quaterniond orientation(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        
        // 在以光学中心为原点的立方体内迭代，寻找可见的未知体素
        for (double dx = -gain_max_range_; dx <= gain_max_range_; dx += octree_->getResolution()) {
            for (double dy = -gain_max_range_; dy <= gain_max_range_; dy += octree_->getResolution()) {
                for (double dz = -gain_max_range_; dz <= gain_max_range_; dz += octree_->getResolution()) {
                    octomap::point3d p(origin.x() + dx, origin.y() + dy, origin.z() + dz);
                    double dist_to_origin = origin.distance(p);

                    // 忽略距离太远或太近的点
                    if (dist_to_origin > gain_max_range_ || dist_to_origin < octree_->getResolution() * 0.5) {
                        continue;
                    }

                    // 视锥剔除：只考虑在相机前方（局部+X方向）的点
                    Eigen::Vector3d point_in_camera_frame = orientation.inverse() * (Eigen::Vector3d(p.x(), p.y(), p.z()) - Eigen::Vector3d(origin.x(), origin.y(), origin.z()));
                    if (point_in_camera_frame.x() < 0.0) {
                        continue;
                    }
                    
                    // 检查体素是否是未知的
                    if (octree_->search(p) == nullptr) {
                        octomap::point3d end_point;
                        // 检查从光学中心到该点的视线是否通畅
                        if (!octree_->castRay(origin, p - origin, end_point, true, dist_to_origin)) {
                            // 如果视线通畅，则累加增益
                            geometric_gain += 1.0;
                            
                            // 查找并累加权重增益
                            double weight = 1.0; // 默认权重为1
                            if (weight_map_) {
                                octomap::OcTreeNode* weight_node = weight_map_->search(p);
                                if (weight_node) {
                                    // getLogOdds() 返回我们存入的float权重
                                    weight = weight_node->getLogOdds();
                                }
                            }
                            weighted_gain += weight;
                        }
                    }
                }
            }
        }
        
        // 将体素数量转换为体积
        double volume_factor = pow(octree_->getResolution(), 3);
        return {geometric_gain * volume_factor, weighted_gain * volume_factor};
    }


    RRTNode selectBestNode() { 
        if (rrt_tree_.empty()) return RRTNode(); // 返回一个无效节点

        RRTNode best_node = rrt_tree_[0]; // 初始最佳节点为根节点
        best_node.utility = -1.0;

        // --- 步骤A: 检查是否有可用的引导目标 ---
        std::vector<Point> current_targets;
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            if (!detected_targets_.empty() )
            // && (ros::Time::now() - last_target_received_time_).toSec() < 10.0) 
            {
                current_targets = detected_targets_;
            }
        }

        // --- 准备探索方向 (可选，但推荐) ---
        // 我们可以简单地用上一个最佳视点和根节点的连线作为探索方向
        Eigen::Vector3d explore_direction(1, 0, 0); // 默认朝前
        if (last_best_pose_valid_) {
            explore_direction.x() = last_best_pose_.position.x - rrt_tree_[0].pose.position.x;
            explore_direction.y() = last_best_pose_.position.y - rrt_tree_[0].pose.position.y;
            explore_direction.z() = last_best_pose_.position.z - rrt_tree_[0].pose.position.z;
            explore_direction.normalize();
        }

        for (auto& node : rrt_tree_) {
            if (node.id == 0) continue; // 不评估根节点

            // --- 计算综合效用 (Utility) ---
            // 这是一个可以任意设计的核心公式
            
            // 1. 路径成本惩罚
            // 避免除以零，如果成本很小，给一个下限
            double cost_penalty = std::max(node.cost_from_root, 0.1);

            // 2. 方向一致性奖励 (可选)
            Eigen::Vector3d node_direction(
                node.pose.position.x - rrt_tree_[0].pose.position.x,
                node.pose.position.y - rrt_tree_[0].pose.position.y,
                node.pose.position.z - rrt_tree_[0].pose.position.z
            );
            node_direction.normalize();
            // 点积越大，方向越一致，奖励越高 (范围-1到1)
            double direction_bonus = explore_direction.dot(node_direction);
            
            // 3. 可操作性评估 (可选，较复杂)
            moveit::core::RobotState state = planning_scene_->getCurrentState();
            double manipulability = 0.0; // 默认可操作性为0

            // --- 【核心修改】应用与 getNBVCallback 中相同的坐标变换 ---
            Eigen::Isometry3d T_world_camera_world_goal;
            tf2::fromMsg(node.pose, T_world_camera_world_goal);
            Eigen::Isometry3d T_world_link8_final = T_world_camera_world_goal * T_camera_world_to_camera_ * T_link8_camera_.inverse();
            geometry_msgs::msg::Pose ik_target_pose_for_link8 = tf2::toMsg(T_world_link8_final);
            
            std::string moveit_ik_frame = "panda_link8";
            if (state.setFromIK(joint_model_group_, ik_target_pose_for_link8, moveit_ik_frame, 0.05)) {
                Eigen::MatrixXd jacobian = state.getJacobian(joint_model_group_);
                Eigen::MatrixXd jj_transpose = jacobian * jacobian.transpose();
                double determinant = jj_transpose.determinant();
                if (determinant > 1e-9) {
                    manipulability = sqrt(determinant);
                }
            }
            
            // --- 步骤B: 【核心修改】新增: 目标导向奖励 (Heuristic Bonus) ---
            double target_bonus = 0.0;
            if (!current_targets.empty()) {
                // 计算从视点位置到最近目标的方向向量
                Eigen::Vector3d vp_pos(node.pose.position.x, node.pose.position.y, node.pose.position.z);
                Eigen::Vector3d closest_target_dir;
                double min_dist_sq = std::numeric_limits<double>::max();

                for (const auto& target : current_targets) {
                    Eigen::Vector3d target_pos(target.x, target.y, target.z);
                    double d_sq = (target_pos - vp_pos).squaredNorm();
                    if (d_sq < min_dist_sq) {
                        min_dist_sq = d_sq;
                        closest_target_dir = target_pos - vp_pos;
                    }
                }
                if (closest_target_dir.norm() > 1e-6) closest_target_dir.normalize();
                
                // 计算视点姿态的朝向 (X轴方向)
                Eigen::Quaterniond vp_q(node.pose.orientation.w, node.pose.orientation.x, node.pose.orientation.y, node.pose.orientation.z);
                Eigen::Vector3d vp_view_dir = vp_q * Eigen::Vector3d::UnitX();
                
                // 点积越大，说明视点越是“正对”着目标，奖励越高 (范围-1到1)
                target_bonus = vp_view_dir.dot(closest_target_dir);

                std::cout << "target_bonus: " << target_bonus << std::endl;
            }


            // --- 定义权重 ---
            double w_geom_gain = 1.0;     // 原始几何增益的权重
            double w_dir = 0.1;           // 探索方向
            double w_manip = 0.0;         // 可操作性
            double w_target_behavior = 0.0; // RRT行为引导的权重
            double w_weight_map = 0.0;    // 权重地图引导的权重

            // --- 计算每个加权项 ---
            // 使用原始几何增益
            double geom_gain_term = (w_geom_gain * node.gain) / cost_penalty;
            
            // 【新增】使用加权地图增益，同样除以成本
            double weight_map_term = (w_weight_map * node.weighted_gain) / cost_penalty;

            double dir_term = w_dir * direction_bonus / cost_penalty;
            double manip_term = w_manip * manipulability / cost_penalty;
            double target_behavior_term = w_target_behavior * std::max(0.0, target_bonus) / cost_penalty;

            // --- 计算总效用 ---
            // 现在总效用是所有项的总和
            node.utility = geom_gain_term + dir_term + manip_term + target_behavior_term + weight_map_term;

            // 更新打印日志，加入新的项
            RCLCPP_INFO(this->get_logger(), "Node[%d] Util: T=%.2f|GeomG=%.2f|MapG=%.2f|Dir=%.2f|Manip=%.2f|TgtBehav=%.2f",
                node.id, node.utility,
                geom_gain_term,
                weight_map_term,
                dir_term,
                manip_term,
                target_behavior_term);
            
            if (node.utility > best_node.utility) {
                best_node = node;
            }
        }

        // 更新上一个最佳视点，为下一次规划做准备
        if (best_node.id != -1) {
            last_best_pose_ = best_node.pose;
            last_best_pose_valid_ = true;
        }

        return best_node;
    
    }


    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> findAndClusterFrontiers() { 
        pcl::PointCloud<pcl::PointXYZ>::Ptr frontiers(new pcl::PointCloud<pcl::PointXYZ>);
        // <<< 核心修复: 在读取地图前加锁 >>>
        std::lock_guard<std::mutex> lock(octomap_mutex_);
        
        if (!octree_) return {};

        // 核心逻辑改变：我们遍历所有“已知-自由”的叶子节点，然后检查它们的邻居
        for (auto it = octree_->begin_leafs(), end = octree_->end_leafs(); it != end; ++it) {
            
            // 条件1: 我们只从已知的、并且是“自由”的体素开始搜索
            if (!octree_->isNodeOccupied(*it)) {
                octomap::point3d p = it.getCoordinate();

                // 检查这个自由体素的26个邻居
                bool is_a_frontier_cell = false;
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dz = -1; dz <= 1; ++dz) {
                            if (dx == 0 && dy == 0 && dz == 0) continue;

                            octomap::point3d neighbor_p = p + octomap::point3d(
                                dx * octree_->getResolution(), 
                                dy * octree_->getResolution(), 
                                dz * octree_->getResolution()
                            );
                            
                            // 查询邻居节点
                            octomap::OcTreeNode* neighbor_node = octree_->search(neighbor_p);
                            
                            // 条件2: 如果邻居节点不存在 (即'search'返回nullptr)，说明它是“未知”空间
                            if (neighbor_node == nullptr) {
                                is_a_frontier_cell = true;
                                break; // 找到了一个未知邻居，就可以确定当前点p是前沿点
                            }
                        }
                        if (is_a_frontier_cell) break;
                    }
                    if (is_a_frontier_cell) break;
                }

                // 如果这个自由体素p的邻居中有未知空间，那么p本身就是一个前沿点
                if (is_a_frontier_cell) {
                    if (p.x() >= min_bound_x_ && p.x() <= max_bound_x_ &&
                    p.y() >= min_bound_y_ && p.y() <= max_bound_y_ &&
                    p.z() >= min_bound_z_ && p.z() <= max_bound_z_)
                    {
                        if (p.z() <= frontier_max_height_) 
                        {
                            frontiers->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
                        }
                    }
                }
            }
        }

        RCLCPP_INFO(this->get_logger(), "Found %zu raw frontier points before clustering.", frontiers->size());

        // --- 后续的可视化和聚类逻辑保持完全不变 ---

        // 发布所有探测到的前沿点，用于可视化
        PointCloud2 frontier_msg;
        pcl::toROSMsg(*frontiers, frontier_msg);
        frontier_msg.header.frame_id = world_frame_;
        frontier_msg.header.stamp = this->get_clock()->now();
        frontier_points_pub_->publish(frontier_msg);

        // PCL聚类
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
        if (frontiers->empty()) return clusters;

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(frontiers);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(cluster_distance_threshold_);
        ec.setMinClusterSize(min_cluster_size_);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(frontiers);
        ec.extract(cluster_indices);

        RCLCPP_INFO(this->get_logger(), "Clustered frontiers into %zu clusters.", cluster_indices.size());

        // 创建带颜色的点云用于可视化验证
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clusters(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (const auto& indices : cluster_indices) {
            int r = rand() % 256, g = rand() % 256, b = rand() % 256;
            for (const auto& idx : indices.indices) {
                pcl::PointXYZRGB point;
                point.x = (*frontiers)[idx].x;
                point.y = (*frontiers)[idx].y;
                point.z = (*frontiers)[idx].z;
                point.r = r; point.g = g; point.b = b;
                colored_clusters->push_back(point);
            }
        }

        PointCloud2 clusters_msg;
        clusters_msg.header.frame_id = world_frame_;
        clusters_msg.header.stamp = this->get_clock()->now();
        frontier_clusters_pub_->publish(clusters_msg);

        // 将无色的簇返回给主逻辑
        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : indices.indices) {
                cluster->push_back((*frontiers)[idx]);
            }
            clusters.push_back(cluster);
        }
        return clusters;
    }

    void publishRRTTree() {
        Marker tree_marker;
        tree_marker.header.frame_id = world_frame_;
        tree_marker.header.stamp = this->get_clock()->now();
        tree_marker.ns = "rrt_tree";
        tree_marker.id = 0;
        tree_marker.type = Marker::LINE_LIST;
        tree_marker.action = Marker::ADD;
        tree_marker.pose.orientation.w = 1.0;
        tree_marker.scale.x = 0.01; // Line width
        tree_marker.color.g = 1.0;  // Green color for the tree
        tree_marker.color.a = 1.0;

        for (const auto& node : rrt_tree_) {
            if (node.parent_id != -1) {
                tree_marker.points.push_back(rrt_tree_[node.parent_id].pose.position);
                tree_marker.points.push_back(node.pose.position);
            }
        }
        rrt_tree_pub_->publish(tree_marker);
    }
    
    void calculateInitialPoseBoundingBox() { 
        RCLCPP_INFO(this->get_logger(), "Calculating bounding box for the robot's initial pose...");

        // 1. 获取一个代表当前机器人状态的对象
        // MoveIt!初始化时，它会从/robot_description/joint_state_map (或默认值) 加载初始姿态
        moveit::core::RobotState initial_state(robot_model_);
        initial_state.setToDefaultValues(); // 确保加载的是SRDF中定义的home或zero姿态

        // (可选) 如果你想使用一个特定的关节角度，可以这样做：
        // std::map<std::string, double> joint_values;
        // joint_values["panda_joint1"] = 0.0;
        // ...
        // initial_state.setVariablePositions(joint_values);

        // 2. 获取规划组的所有连杆名称
        const std::vector<std::string>& link_names = joint_model_group_->getLinkModelNames();

        RCLCPP_INFO(this->get_logger(), "Calculating bounding box for links: ");
        for(const auto& name : link_names) {
            std::cout << "- " << name << std::endl;
        }

        // 3. 初始化极值
        robot_min_x_ = robot_min_y_ = robot_min_z_ = std::numeric_limits<double>::max();
        robot_max_x_ = robot_max_y_ = robot_max_z_ = -std::numeric_limits<double>::max();

        // 4. 遍历所有连杆，进行正向运动学并更新包围盒
        for (const std::string& link_name : link_names) {
            // 使用正向运动学，计算出该连杆在世界坐标系下的位姿
            const Eigen::Isometry3d& link_pose = initial_state.getGlobalLinkTransform(link_name);
            
            // 提取位置信息
            double x = link_pose.translation().x();
            double y = link_pose.translation().y();
            double z = link_pose.translation().z();

            // 更新六个极值
            if (x < robot_min_x_) robot_min_x_ = x;
            if (x > robot_max_x_) robot_max_x_ = x;
            if (y < robot_min_y_) robot_min_y_ = y;
            if (y > robot_max_y_) robot_max_y_ = y;
            if (z < robot_min_z_) robot_min_z_ = z;
            if (z > robot_max_z_) robot_max_z_ = z;
        }

        // 5. (重要) 考虑连杆的体积
        // 上面的计算只考虑了每个连杆原点(origin)的位置，没有考虑连杆本身的尺寸。
        // 一个简单的、保守的近似是给包围盒增加一个安全余量。
        double safety_margin = 0.15; // 15cm，这个值应该大于你最胖的连杆半径
        robot_min_x_ -= safety_margin; robot_max_x_ += safety_margin;
        robot_min_y_ -= safety_margin; robot_max_y_ += safety_margin;
        robot_min_z_ -= safety_margin; robot_max_z_ += safety_margin;

        RCLCPP_INFO(this->get_logger(), "Static robot bounding box calculated: X[%.2f, %.2f], Y[%.2f, %.2f], Z[%.2f, %.2f]",
            robot_min_x_, robot_max_x_, robot_min_y_, robot_max_y_, robot_min_z_, robot_max_z_);
    }

    // --- 成员变量 ---
    // --- 【已修正】补全所有缺失的成员变量 ---
    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Service<GetNBV>::SharedPtr get_nbv_service_;
    rclcpp::Service<GetInitialCoverage>::SharedPtr get_coverage_service_;
    rclcpp::Service<UpdateWeights>::SharedPtr update_weights_service_;
    rclcpp::Publisher<Marker>::SharedPtr rrt_tree_pub_;
    rclcpp::Publisher<PoseStamped>::SharedPtr best_viewpoint_pub_;
    rclcpp::Publisher<PointCloud2>::SharedPtr frontier_points_pub_;
    rclcpp::Publisher<PointCloud2>::SharedPtr frontier_clusters_pub_;
    rclcpp::Subscription<PoseArray>::SharedPtr target_sub_;

    std::mutex octomap_mutex_;
    std::shared_ptr<octomap::OcTree> octree_;
    std::shared_ptr<octomap::OcTree> weight_map_; 
    bool map_received_ = false;
    robot_model_loader::RobotModelLoaderPtr robot_model_loader_;
    moveit::core::RobotModelPtr robot_model_;
    planning_scene::PlanningScenePtr planning_scene_;
    const moveit::core::JointModelGroup* joint_model_group_;
    std::string world_frame_, planning_group_, camera_link_name_;
    double cluster_distance_threshold_, generation_radius_, gain_max_range_, frontier_max_height_, target_coverage_radius_;
    int min_cluster_size_;
    double min_bound_x_, max_bound_x_, min_bound_y_, max_bound_y_, min_bound_z_, max_bound_z_;
    double robot_min_x_, robot_max_x_, robot_min_y_, robot_max_y_, robot_min_z_, robot_max_z_;
    bool use_static_robot_box_;
    int rrt_max_iterations_;
    double rrt_extension_range_;
    std::vector<RRTNode> rrt_tree_;
    bool last_best_pose_valid_ = false;
    Pose last_best_pose_; // 【已修正】使用别名
    std::vector<Point> detected_targets_; // 【已修正】使用别名
    rclcpp::Time last_target_received_time_;
    std::mutex target_mutex_;
    double target_bias_probability_, target_bonus_weight_;
    std::mt19937 random_generator_;
    std::uniform_real_distribution<double> uniform_dist_;
    int run_id, level, scene;
    std::shared_ptr<StatisticsManager> stats_manager_;
    Point target_to_find_; // 【已修正】使用别名

    // TF2缓冲和监听器
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    Eigen::Isometry3d T_link8_camera_;
    Eigen::Isometry3d T_camera_world_to_camera_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    // 1. 创建节点实例
    auto rrt_node = std::make_shared<RRTExplorerServer>();

    // 2. 【核心修改】调用init()函数并检查其返回值
    if (!rrt_node->init()) {
        RCLCPP_ERROR(rrt_node->get_logger(), "Failed to initialize RRT Explorer Node. Shutting down.");
        rclcpp::shutdown();
        return -1; // 返回一个错误码
    }

    // 3. 【核心修改】使用多线程执行器来spin
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(rrt_node);
    
    // 阻塞并处理回调
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
