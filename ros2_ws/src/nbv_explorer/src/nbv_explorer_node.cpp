#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <std_msgs/msg/float64.hpp>

#include <octomap_msgs/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <octomap/octomap.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <Eigen/Geometry>

#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/planning_scene/planning_scene.hpp>

#include <mutex>
#include <chrono>
#include <random>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <cstdlib>

#include "nbv_explorer/srv/get_nbv.hpp"
#include "nbv_explorer/srv/get_initial_coverage.hpp"
#include "nbv_explorer/srv/update_weights.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;
using std::placeholders::_2;
namespace fs = std::filesystem;


struct Viewpoint {
    geometry_msgs::msg::Pose pose;
    double gain;
};


class StatisticsManager {
public:
    StatisticsManager(rclcpp::Node* node_ptr, int _id, int _level, int _scene) 
    : node_(node_ptr) 
    {
        const char * experiments_env = std::getenv("COMPASS_EXPERIMENTS_DIR");
        const char * root_env = std::getenv("COMPASS_ROOT");
        fs::path experiments_root = experiments_env
            ? fs::path(experiments_env)
            : (root_env ? fs::path(root_env) / "experiments" : fs::path("experiments"));
        std::string method_path = (experiments_root / "single_nbv" / ("level" + std::to_string(_level)) / ("scene_" + std::to_string(_scene))).string();

        result_path_ = fs::path(method_path) / ("run_" + std::to_string(_id));
        try {
            fs::create_directories(result_path_);
        } catch (const fs::filesystem_error& e) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to create run directory %s: %s", result_path_.c_str(), e.what());
            return;
        }

        explored_volume_pub_ = node_->create_publisher<std_msgs::msg::Float64>("/exploration/explored_volume", 10);
        path_length_pub_ = node_->create_publisher<std_msgs::msg::Float64>("/exploration/path_length", 10);
        
        volume_log_.open(result_path_ / "explored_volume_vs_time.csv", std::ofstream::out | std::ofstream::trunc);
        path_log_.open(result_path_ / "path_length_vs_time.csv", std::ofstream::out | std::ofstream::trunc);
        manipulability_log_.open(result_path_ / "manipulability_vs_time.csv", std::ofstream::out | std::ofstream::trunc);
        target_coverage_log_.open(result_path_ / "target_coverage_vs_time.csv", std::ofstream::out | std::ofstream::trunc);

        volume_log_ << "timestamp,elapsed_time,volume_m3" << std::endl;
        path_log_ << "timestamp,elapsed_time,path_length_m" << std::endl;
        manipulability_log_ << "timestamp,elapsed_time,manipulability" << std::endl;
        target_coverage_log_ << "timestamp,elapsed_time,coverage_ratio" << std::endl;

        start_time_ = node_->get_clock()->now();
    }

    ~StatisticsManager() {
        if (volume_log_.is_open()) volume_log_.close();
        if (path_log_.is_open()) path_log_.close();
        if (manipulability_log_.is_open()) manipulability_log_.close();
        if (target_coverage_log_.is_open()) target_coverage_log_.close();
    }

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
        
        auto msg = std_msgs::msg::Float64();
        msg.data = current_volume;
        explored_volume_pub_->publish(msg);
        volume_log_ << now.nanoseconds() << "," << elapsed_time << "," << current_volume << std::endl;
    }

    void updatePathLength(const geometry_msgs::msg::Point& new_position) {
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

        auto msg = std_msgs::msg::Float64();
        msg.data = total_path_length_;
        path_length_pub_->publish(msg);
        path_log_ << now.nanoseconds() << "," << elapsed_time << "," << total_path_length_ << std::endl;
    }

    void updateManipulability(const moveit::core::RobotState& current_state,
            const moveit::core::JointModelGroup* joint_model_group)
    {
        if (!manipulability_log_.is_open() || !joint_model_group) return;

        Eigen::MatrixXd jacobian = current_state.getJacobian(joint_model_group);
        Eigen::MatrixXd jj_transpose = jacobian * jacobian.transpose();
        double determinant = jj_transpose.determinant();

        double manipulability = 0.0;
        if (determinant > 1e-9) {
            manipulability = sqrt(determinant);
        }

        rclcpp::Time now = node_->get_clock()->now();
        double elapsed_time = (now - start_time_).seconds();
        manipulability_log_ << now.nanoseconds() << "," << elapsed_time << "," << manipulability << std::endl;
    }
    
    void checkTargetVisibility(const octomap::OcTree* octree, const geometry_msgs::msg::Pose& camera_pose, const geometry_msgs::msg::Point& target_point) {
        if (!octree || target_found_ || !path_log_.is_open()) {
            std::cout << "ggggggggggggggggggggggggg" << std::endl;
            return;
        }

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
            
            std::ofstream target_log(result_path_ / "target_found_log.csv", std::ofstream::out | std::ofstream::trunc);
            target_log << "timestamp,elapsed_time,path_length_m" << std::endl;
            target_log << now.nanoseconds() << "," << elapsed_time << "," << total_path_length_ << std::endl;
            target_log.close();

            target_found_ = true;
        }
    }

    void updateTargetCoverage(const octomap::OcTree* octree, 
            const geometry_msgs::msg::Point& target_point,
            double coverage_radius)
    {
        if (!octree || target_found_ || !target_coverage_log_.is_open()) return;

        int total_voxels = 0;
        int known_voxels = 0;
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
    geometry_msgs::msg::Point last_position_;
    double total_path_length_ = 0.0;
    bool target_found_ = false;
    std::ofstream volume_log_;
    std::ofstream path_log_;
    std::ofstream manipulability_log_;
    std::ofstream target_coverage_log_;
    fs::path result_path_;
    double time_to_find_target_ = -1.0;
    double path_at_target_found_ = -1.0;
};


struct RRTNode {
    int id;
    geometry_msgs::msg::Pose pose;
    double gain = 0.0;
    double weighted_gain = 0.0;
    int parent_id = -1;
    double cost_from_root = 0.0;
    double utility = 0.0;
};

class RRTExplorerServer : public rclcpp::Node {
public:
    RRTExplorerServer() : Node("nbv_explorer_node"), random_generator_(std::random_device{}())
    {
        RCLCPP_INFO(this->get_logger(), "Constructing NBV Explorer Node...");
    }

    void init()
    {
        RCLCPP_INFO(this->get_logger(), "Initializing NBV Explorer Node...");
        
        this->declare_parameter<std::string>("world_frame", "panda_link0");
        this->declare_parameter<std::string>("planning_group", "panda_arm");
        this->declare_parameter<std::string>("camera_link_name", "realsense_camera_world");
        this->declare_parameter("frontier_cluster_distance", 0.5);
        this->declare_parameter("viewpoint_generation_radius", 0.8);
        this->declare_parameter("min_cluster_size", 10);
        this->declare_parameter("viewpoint_samples_per_cluster", 20);
        this->declare_parameter("gain_max_range", 3.0);
        this->declare_parameter("rrt/max_iterations", 200);
        this->declare_parameter("rrt/extension_range", 0.3);
        this->declare_parameter("frontier_max_height", 1.2);
        this->declare_parameter("target/coverage_radius", 0.5);
        this->declare_parameter("bounds/min_x", -1.0);
        this->declare_parameter("bounds/max_x", 1.0);
        this->declare_parameter("bounds/min_y", -1.0);
        this->declare_parameter("bounds/max_y", 1.0);
        this->declare_parameter("bounds/min_z", 0.15);
        this->declare_parameter("bounds/max_z", 1.0);
        this->declare_parameter("target/x", 100.0);
        this->declare_parameter("target/y", 100.0);
        this->declare_parameter("target/z", 100.0);
        this->declare_parameter("use_static_robot_box", true);
        this->declare_parameter("guidance/target_bias_probability", 0.5); 
        this->declare_parameter("guidance/target_bonus_weight", 2.0);
        this->declare_parameter("run_id", 0);
        this->declare_parameter("level", 1);
        this->declare_parameter("scene", 1);
        
        this->get_parameter("world_frame", world_frame_);
        this->get_parameter("planning_group", planning_group_);
        this->get_parameter("camera_link_name", camera_link_name_);
        this->get_parameter("frontier_cluster_distance", cluster_distance_threshold_);
        this->get_parameter("viewpoint_generation_radius", generation_radius_);
        this->get_parameter("min_cluster_size", min_cluster_size_);
        this->get_parameter("viewpoint_samples_per_cluster", samples_per_cluster_);
        this->get_parameter("gain_max_range", gain_max_range_);
        this->get_parameter("rrt/max_iterations", rrt_max_iterations_);
        this->get_parameter("rrt/extension_range", rrt_extension_range_);
        this->get_parameter("frontier_max_height", frontier_max_height_);
        this->get_parameter("target/coverage_radius", target_coverage_radius_);
        this->get_parameter("bounds/min_x", min_bound_x_);
        this->get_parameter("bounds/max_x", max_bound_x_);
        this->get_parameter("bounds/min_y", min_bound_y_);
        this->get_parameter("bounds/max_y", max_bound_y_);
        this->get_parameter("bounds/min_z", min_bound_z_);
        this->get_parameter("bounds/max_z", max_bound_z_);
        this->get_parameter("target/x", target_to_find_.x);
        this->get_parameter("target/y", target_to_find_.y);
        this->get_parameter("target/z", target_to_find_.z);
        this->get_parameter("use_static_robot_box", use_static_robot_box_);
        this->get_parameter("guidance/target_bias_probability", target_bias_probability_);
        this->get_parameter("guidance/target_bonus_weight", target_bonus_weight_);
        this->get_parameter("run_id", run_id);
        this->get_parameter("level", level);
        this->get_parameter("scene", scene);

        T_link8_camera_ = Eigen::Isometry3d::Identity();
        T_link8_camera_.matrix()(0, 0) = 0.707; T_link8_camera_.matrix()(0, 1) = -0.707; T_link8_camera_.matrix()(0, 2) = 0.0;
        T_link8_camera_.matrix()(1, 0) = 0.707; T_link8_camera_.matrix()(1, 1) = 0.707;  T_link8_camera_.matrix()(1, 2) = 0.0;
        T_link8_camera_.matrix()(2, 0) = 0.0;   T_link8_camera_.matrix()(2, 1) = 0.0;    T_link8_camera_.matrix()(2, 2) = 1.0;
        T_link8_camera_.translation() = Eigen::Vector3d(0.035, -0.035, 0.050);
        RCLCPP_INFO(this->get_logger(), "Hardcoded link8->camera transform initialized.");
        
        T_camera_world_to_camera_ = Eigen::Isometry3d::Identity();
        Eigen::Matrix3d rotation_matrix;
        rotation_matrix << 0.0, 0.0, 1.0,
                           -1.0, 0.0, 0.0,
                           0.0, -1.0, 0.0;
        T_camera_world_to_camera_.linear() = rotation_matrix;
        T_camera_world_to_camera_.translation() = Eigen::Vector3d(0.0, 0.0, 0.0);
        RCLCPP_INFO(this->get_logger(), "Hardcoded camera_world->camera transform initialized.");

        uniform_dist_ = std::uniform_real_distribution<double>(0.0, 1.0);

        robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(shared_from_this(), "robot_description");
        robot_model_ = robot_model_loader_->getModel();
        planning_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_model_);
        joint_model_group_ = robot_model_->getJointModelGroup(planning_group_);
        if (!joint_model_group_) {
            RCLCPP_FATAL(this->get_logger(), "Planning group '%s' not found.", planning_group_.c_str());
            rclcpp::shutdown(); return;
        }

        if (use_static_robot_box_) {
            timer_ = this->create_wall_timer(500ms, [this]() {
                this->timer_->cancel();
                this->calculateInitialPoseBoundingBox();
            });
        }

        octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
            "/octomap_full", 1, std::bind(&RRTExplorerServer::octomapCallback, this, _1));
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&RRTExplorerServer::jointStateCallback, this, _1));
        target_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/potential_targets", 10, std::bind(&RRTExplorerServer::targetCallback, this, _1));
            
        get_nbv_service_ = this->create_service<nbv_explorer::srv::GetNBV>(
            "get_next_best_viewpoint", std::bind(&RRTExplorerServer::getNBVCallback, this, _1, _2));
        get_coverage_service_ = this->create_service<nbv_explorer::srv::GetInitialCoverage>(
            "get_coverage_ratio", std::bind(&RRTExplorerServer::getCoverageCallback, this, _1, _2));
        update_weights_service_ = this->create_service<nbv_explorer::srv::UpdateWeights>(
            "update_exploration_weights", std::bind(&RRTExplorerServer::updateWeightsCallback, this, _1, _2));

        rrt_tree_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_tree_vis", 10);
        best_viewpoint_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/best_viewpoint", 1);
        frontier_points_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/frontier_points", 1);
        frontier_clusters_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/frontier_clusters", 1);
        candidate_viewpoints_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/candidate_viewpoints", 1);
        raw_viewpoints_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/raw_viewpoints", 1);

        stats_manager_ = std::make_shared<StatisticsManager>(this, run_id, level, scene);
        
        RCLCPP_INFO(this->get_logger(), "NBV Explorer Server initialized. Ready to provide NBVs.");
    }

private:
    void getNBVCallback(const std::shared_ptr<nbv_explorer::srv::GetNBV::Request> req,
                        std::shared_ptr<nbv_explorer::srv::GetNBV::Response> res)
    {
        (void)req;
        if (!map_received_) {
            RCLCPP_ERROR(this->get_logger(), "Cannot provide NBV: No OctoMap has been received yet.");
            res->success = false;
            res->message = "No OctoMap received.";
            return;
        }

        RCLCPP_INFO(this->get_logger(), "--- Received NBV Request. Starting selection cycle. ---");

        auto clusters = findAndClusterFrontiers();
        if (clusters.empty()) {
            res->success = false;
            res->message = "No frontiers found.";
            RCLCPP_WARN(this->get_logger(), "NBV selection failed: %s", res->message.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Found %zu frontier clusters.", clusters.size());
        
        auto viewpoints = generateAndFilterViewpoints(clusters);
        if (viewpoints.empty()) {
            res->success = false;
            res->message = "No valid (IK solvable and collision-free) viewpoints found.";
            RCLCPP_WARN(this->get_logger(), "NBV selection failed: %s", res->message.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Generated %zu valid candidate viewpoints.", viewpoints.size());

        Viewpoint best_viewpoint = evaluateViewpoints(viewpoints);

        if (best_viewpoint.gain <= 0.0) {
            res->success = false;
            res->message = "No viewpoint with positive gain found.";
            RCLCPP_WARN(this->get_logger(), "NBV selection failed: %s", res->message.c_str());
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Best viewpoint selected with gain %.2f.", best_viewpoint.gain);

        res->nbv.header.frame_id = world_frame_;
        res->nbv.header.stamp = this->get_clock()->now();
        res->nbv.pose = best_viewpoint.pose;
        res->success = true;
        res->message = "Successfully found a Next-Best-Viewpoint.";

        auto best_vp_msg = std::make_unique<geometry_msgs::msg::PoseStamped>();
        best_vp_msg->header = res->nbv.header;
        best_vp_msg->pose = res->nbv.pose;
        best_viewpoint_pub_->publish(std::move(best_vp_msg));
        
        RCLCPP_INFO(this->get_logger(), "\033[1;32mSuccessfully provided a Next-Best-Viewpoint.\033[0m");
    }

    void getCoverageCallback(const std::shared_ptr<nbv_explorer::srv::GetInitialCoverage::Request> req,
                             std::shared_ptr<nbv_explorer::srv::GetInitialCoverage::Response> res)
    {
        (void)req;
        std::lock_guard<std::mutex> lock(octomap_mutex_);
        if (!octree_) {
            res->coverage_ratio = 0.0;
            return;
        }

        if (!use_static_robot_box_) {
            res->coverage_ratio = 1.0;
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
    
    void updateWeightsCallback(const std::shared_ptr<nbv_explorer::srv::UpdateWeights::Request> req,
                               std::shared_ptr<nbv_explorer::srv::UpdateWeights::Response> res)
    {
        std::lock_guard<std::mutex> lock(octomap_mutex_);
        if(octree_ && (!weight_map_)){ 
            weight_map_ = std::make_shared<octomap::OcTree>(octree_->getResolution());
        }

        weight_map_->clear();
        pcl::PointCloud<pcl::PointXYZI>::Ptr weighted_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(req->weighted_points, *weighted_cloud);

        for (const auto& point : *weighted_cloud) {
            if (point.intensity >= 0) {
                weight_map_->updateNode(point.x, point.y, point.z, static_cast<float>(point.intensity));
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Updated weight map with %zu points.", weighted_cloud->size());
        res->success = true;
    }
    
    void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(octomap_mutex_);
        octomap::AbstractOcTree* tree = octomap_msgs::fullMsgToMap(*msg);
        if (tree) {
            octree_.reset(dynamic_cast<octomap::OcTree*>(tree));
            if(!map_received_) RCLCPP_INFO(this->get_logger(), "First OctoMap received!");
            map_received_ = true;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to deserialize OctoMap message.");
        }
        if (stats_manager_) {
            stats_manager_->updateExploredVolume(octree_.get());
        }
    }

    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(octomap_mutex_);
        if (planning_scene_) {
            moveit::core::RobotState& current_state = planning_scene_->getCurrentStateNonConst();
            current_state.setVariablePositions(msg->name, msg->position);
            if (stats_manager_ && current_state.knowsFrameTransform(camera_link_name_)) {
                const Eigen::Isometry3d& transform = current_state.getGlobalLinkTransform(camera_link_name_);
                geometry_msgs::msg::Pose current_camera_pose = tf2::toMsg(transform);
                
                stats_manager_->updatePathLength(current_camera_pose.position);
                stats_manager_->checkTargetVisibility(octree_.get(), current_camera_pose, target_to_find_);
                stats_manager_->updateManipulability(current_state, joint_model_group_);
                stats_manager_->updateTargetCoverage(octree_.get(), target_to_find_, target_coverage_radius_);
            }
        }
    }

    void targetCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(target_mutex_);
        detected_targets_.clear();
        for(const auto& pose : msg->poses) {
            detected_targets_.push_back(pose.position);
        }
        last_target_received_time_ = this->get_clock()->now();
        if (!detected_targets_.empty()) {
            RCLCPP_INFO(this->get_logger(), "\033[1;35mReceived %zu heuristic targets. Exploration is now target-driven.\033[0m", 
                    detected_targets_.size());
        }
    }

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> findAndClusterFrontiers();
    std::vector<Viewpoint> generateAndFilterViewpoints(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters);
    Viewpoint evaluateViewpoints(std::vector<Viewpoint>& viewpoints);
    void calculateInitialPoseBoundingBox();

    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr target_sub_;
    rclcpp::Service<nbv_explorer::srv::GetNBV>::SharedPtr get_nbv_service_;
    rclcpp::Service<nbv_explorer::srv::GetInitialCoverage>::SharedPtr get_coverage_service_;
    rclcpp::Service<nbv_explorer::srv::UpdateWeights>::SharedPtr update_weights_service_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr rrt_tree_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr best_viewpoint_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frontier_points_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frontier_clusters_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr candidate_viewpoints_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr raw_viewpoints_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::mutex octomap_mutex_;
    std::mutex target_mutex_;

    std::shared_ptr<octomap::OcTree> octree_;
    std::shared_ptr<octomap::OcTree> weight_map_;
    bool map_received_ = false;

    robot_model_loader::RobotModelLoaderPtr robot_model_loader_;
    moveit::core::RobotModelPtr robot_model_;
    planning_scene::PlanningScenePtr planning_scene_;
    const moveit::core::JointModelGroup* joint_model_group_;

    std::string world_frame_;
    std::string planning_group_;
    std::string camera_link_name_;
    double cluster_distance_threshold_;
    double generation_radius_;
    int min_cluster_size_;
    int samples_per_cluster_;
    double gain_max_range_;
    double frontier_max_height_;
    double target_coverage_radius_;
    double min_bound_x_, max_bound_x_;
    double min_bound_y_, max_bound_y_;
    double min_bound_z_, max_bound_z_;
    double robot_min_x_, robot_max_x_;
    double robot_min_y_, robot_max_y_;
    double robot_min_z_, robot_max_z_;
    bool use_static_robot_box_;
    int rrt_max_iterations_;
    double rrt_extension_range_;
    double target_bias_probability_;
    double target_bonus_weight_;
    int run_id, level, scene;
    geometry_msgs::msg::Point target_to_find_;

    std::vector<RRTNode> rrt_tree_;
    bool last_best_pose_valid_ = false;
    geometry_msgs::msg::Pose last_best_pose_;
    
    std::vector<geometry_msgs::msg::Point> detected_targets_;
    rclcpp::Time last_target_received_time_{0, 0, RCL_ROS_TIME};

    std::mt19937 random_generator_;
    std::uniform_real_distribution<double> uniform_dist_;
    
    std::shared_ptr<StatisticsManager> stats_manager_;

    Eigen::Isometry3d T_link8_camera_;
    Eigen::Isometry3d T_camera_world_to_camera_;
};

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> RRTExplorerServer::findAndClusterFrontiers() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr frontiers(new pcl::PointCloud<pcl::PointXYZ>);
    std::lock_guard<std::mutex> lock(octomap_mutex_);
    if (!octree_) return {};

    for (auto it = octree_->begin_leafs(), end = octree_->end_leafs(); it != end; ++it) {
        if (!octree_->isNodeOccupied(*it)) {
            octomap::point3d p = it.getCoordinate();
            bool is_a_frontier_cell = false;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        octomap::point3d neighbor_p = p + octomap::point3d(dx * octree_->getResolution(), dy * octree_->getResolution(), dz * octree_->getResolution());
                        if (octree_->search(neighbor_p) == nullptr) {
                            is_a_frontier_cell = true;
                            break;
                        }
                    }
                    if (is_a_frontier_cell) break;
                }
                if (is_a_frontier_cell) break;
            }
            if (is_a_frontier_cell) {
                if (p.x() >= min_bound_x_ && p.x() <= max_bound_x_ && p.y() >= min_bound_y_ && p.y() <= max_bound_y_ && p.z() >= min_bound_z_ && p.z() <= max_bound_z_) {
                    frontiers->push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
                }
            }
        }
    }

    auto frontier_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*frontiers, *frontier_msg);
    frontier_msg->header.frame_id = world_frame_;
    frontier_msg->header.stamp = this->get_clock()->now();
    frontier_points_pub_->publish(std::move(frontier_msg));

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

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clusters(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& indices : cluster_indices) {
        int r = rand() % 256, g = rand() % 256, b = rand() % 256;
        for (const auto& idx : indices.indices) {
            pcl::PointXYZRGB point;
            point.x = (*frontiers)[idx].x; point.y = (*frontiers)[idx].y; point.z = (*frontiers)[idx].z;
            point.r = r; point.g = g; point.b = b;
            colored_clusters->push_back(point);
        }
    }
    auto clusters_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*colored_clusters, *clusters_msg);
    clusters_msg->header.frame_id = world_frame_;
    clusters_msg->header.stamp = this->get_clock()->now();
    frontier_clusters_pub_->publish(std::move(clusters_msg));

    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : indices.indices) {
            cluster->push_back((*frontiers)[idx]);
        }
        clusters.push_back(cluster);
    }
    return clusters;
}
    
std::vector<Viewpoint> RRTExplorerServer::generateAndFilterViewpoints(
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters)
{
    std::vector<Viewpoint> valid_viewpoints;
    auto raw_viewpoints_msg = std::make_unique<geometry_msgs::msg::PoseArray>();
    raw_viewpoints_msg->header.frame_id = world_frame_;

    moveit::core::RobotState seed_state = planning_scene_->getCurrentState();
    const std::string moveit_ik_frame = "panda_link8";
    
    octomap::point3d robot_base_center(0.3, 0.0, 0.0);

    for (const auto& cluster : clusters) {
        if (cluster->empty()) continue;
        
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cluster, centroid);
        octomap::point3d cluster_center(centroid[0], centroid[1], centroid[2]);

        for (int i = 0; i < samples_per_cluster_; ++i) {
            double phi = acos(1 - 2 * uniform_dist_(random_generator_));
            double theta = 2 * M_PI * uniform_dist_(random_generator_);
            
            geometry_msgs::msg::Pose candidate_camera_pose;
            candidate_camera_pose.position.x = robot_base_center.x() + generation_radius_ * sin(phi) * cos(theta);
            candidate_camera_pose.position.y = robot_base_center.y() + generation_radius_ * sin(phi) * sin(theta);
            candidate_camera_pose.position.z = robot_base_center.z() + generation_radius_ * cos(phi);

            Eigen::Vector3d view_direction(cluster_center.x() - candidate_camera_pose.position.x,
                                        cluster_center.y() - candidate_camera_pose.position.y,
                                        cluster_center.z() - candidate_camera_pose.position.z);
            view_direction.normalize();
            Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitX(), view_direction);
            candidate_camera_pose.orientation = tf2::toMsg(q);

            raw_viewpoints_msg->poses.push_back(candidate_camera_pose);

            if (candidate_camera_pose.position.z < min_bound_z_ || candidate_camera_pose.position.z > max_bound_z_ ||
                candidate_camera_pose.position.x < min_bound_x_ || candidate_camera_pose.position.x > max_bound_x_ ||
                candidate_camera_pose.position.y < min_bound_y_ || candidate_camera_pose.position.y > max_bound_y_)
            {
                continue;
            }

            Eigen::Isometry3d T_world_camera_world_goal;
            tf2::fromMsg(candidate_camera_pose, T_world_camera_world_goal);

            Eigen::Isometry3d T_world_link8_goal = T_world_camera_world_goal * T_camera_world_to_camera_ * T_link8_camera_.inverse();

            geometry_msgs::msg::Pose ik_target_pose_for_link8 = tf2::toMsg(T_world_link8_goal);


            if (seed_state.setFromIK(joint_model_group_, ik_target_pose_for_link8, moveit_ik_frame, 0.1)) 
            {
                if (!planning_scene_->isStateColliding(seed_state, joint_model_group_->getName())) {
                    valid_viewpoints.push_back({candidate_camera_pose, 0.0});
                }
            }
        }
    }

    raw_viewpoints_msg->header.stamp = this->get_clock()->now();
    raw_viewpoints_pub_->publish(std::move(raw_viewpoints_msg));

    auto viewpoints_msg = std::make_unique<geometry_msgs::msg::PoseArray>();
    viewpoints_msg->header.frame_id = world_frame_;
    viewpoints_msg->header.stamp = this->get_clock()->now();
    for (const auto& vp : valid_viewpoints) {
        viewpoints_msg->poses.push_back(vp.pose);
    }
    candidate_viewpoints_pub_->publish(std::move(viewpoints_msg));

    return valid_viewpoints;
}

Viewpoint RRTExplorerServer::evaluateViewpoints(std::vector<Viewpoint>& viewpoints) {
    std::lock_guard<std::mutex> lock(octomap_mutex_);
    Viewpoint best_vp;
    best_vp.gain = -1.0;
    if (!octree_) return best_vp;
    
    const double resolution = octree_->getResolution();
    int viewpoint_count = 0;

    for (auto& vp : viewpoints) {
        auto single_vp_start_time = std::chrono::steady_clock::now();
        RCLCPP_INFO(this->get_logger(), "Evaluating viewpoint %d/%zu...", ++viewpoint_count, viewpoints.size());

        double current_gain = 0.0;
        octomap::point3d origin(vp.pose.position.x, vp.pose.position.y, vp.pose.position.z);
        Eigen::Quaterniond orientation;
        tf2::fromMsg(vp.pose.orientation, orientation);

        for (double dx = -gain_max_range_; dx <= gain_max_range_; dx += resolution) {
            for (double dy = -gain_max_range_; dy <= gain_max_range_; dy += resolution) {
                for (double dz = -gain_max_range_; dz <= gain_max_range_; dz += resolution) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    octomap::point3d p(origin.x() + dx, origin.y() + dy, origin.z() + dz);
                    if (origin.distance(p) > gain_max_range_) continue;
                    if (origin.distance(p) < resolution * 0.5) continue;

                    Eigen::Vector3d point_vec(p.x() - origin.x(), p.y() - origin.y(), p.z() - origin.z());
                    Eigen::Vector3d point_in_camera_frame = orientation.inverse() * point_vec;
                    if (point_in_camera_frame.x() < 0) continue;

                    if (octree_->search(p) == nullptr) {
                        octomap::point3d end_point;
                        if (!octree_->castRay(origin, p - origin, end_point, true, origin.distance(p))) {
                            current_gain += 1.0; // gain_unknown
                        }
                    }
                }
            }
        }
        vp.gain = current_gain; 
        if (vp.gain > best_vp.gain) {
            best_vp = vp;
        }
        auto single_vp_end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> single_vp_duration = single_vp_end_time - single_vp_start_time;
        RCLCPP_INFO(this->get_logger(), "  - Viewpoint %d/%zu evaluated. Gain: %.1f, Time: %.2f ms", viewpoint_count, viewpoints.size(), vp.gain, single_vp_duration.count());
    }
    if (best_vp.gain > 0) {
        best_vp.gain *= pow(resolution, 3);
    }
    RCLCPP_INFO(this->get_logger(), "Exiting evaluateViewpoints. Best gain: %.2f", best_vp.gain);
    return best_vp;
}

void RRTExplorerServer::calculateInitialPoseBoundingBox() {
    RCLCPP_INFO(this->get_logger(), "Calculating bounding box for the robot's initial pose...");
    moveit::core::RobotState initial_state(robot_model_);
    initial_state.setToDefaultValues();
    const std::vector<std::string>& link_names = joint_model_group_->getLinkModelNames();
    RCLCPP_INFO(this->get_logger(), "Calculating bounding box for links of group '%s'", planning_group_.c_str());
    
    for(const auto& name : link_names) {
        RCLCPP_INFO(this->get_logger(), "- %s", name.c_str());
    }

    robot_min_x_ = robot_min_y_ = robot_min_z_ = std::numeric_limits<double>::max();
    robot_max_x_ = robot_max_y_ = robot_max_z_ = -std::numeric_limits<double>::max();

    for (const std::string& link_name : link_names) {
        const Eigen::Isometry3d& link_pose = initial_state.getGlobalLinkTransform(link_name);
        double x = link_pose.translation().x();
        double y = link_pose.translation().y();
        double z = link_pose.translation().z();
        if (x < robot_min_x_) robot_min_x_ = x;
        if (x > robot_max_x_) robot_max_x_ = x;
        if (y < robot_min_y_) robot_min_y_ = y;
        if (y > robot_max_y_) robot_max_y_ = y;
        if (z < robot_min_z_) robot_min_z_ = z;
        if (z > robot_max_z_) robot_max_z_ = z;
    }

    double safety_margin = 0.15;
    robot_min_x_ -= safety_margin; robot_max_x_ += safety_margin;
    robot_min_y_ -= safety_margin; robot_max_y_ += safety_margin;
    robot_min_z_ -= safety_margin; robot_max_z_ += safety_margin;

    RCLCPP_INFO(this->get_logger(), "Static robot bounding box calculated: X[%.2f, %.2f], Y[%.2f, %.2f], Z[%.2f, %.2f]",
            robot_min_x_, robot_max_x_, robot_min_y_, robot_max_y_, robot_min_z_, robot_max_z_);
}


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RRTExplorerServer>();
    node->init();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
