#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include <octomap/OcTree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <memory>
#include "rcl_interfaces/msg/set_parameters_result.hpp"

class OctomapBuilder : public rclcpp::Node {
public:
    explicit OctomapBuilder(const rclcpp::NodeOptions & options) 
        : rclcpp::Node("octomap_builder_node", options), tree_(0.05) {
        
        this->declare_parameter<std::string>("world_frame", "world");
        this->declare_parameter<double>("resolution", 0.05);
        this->declare_parameter<double>("max_range", 5.0);
        this->declare_parameter<double>("min_range", 0.1);
        this->declare_parameter<double>("voxel_filter_size", 0.01);
        this->declare_parameter<std::string>("cloud_in_topic", "/camera_pointcloud");

        this->get_parameter("world_frame", world_frame_);
        this->get_parameter("resolution", resolution_);
        this->get_parameter("max_range", max_range_);
        this->get_parameter("min_range", min_range_);
        this->get_parameter("voxel_filter_size", voxel_size_);
        
        std::string cloud_topic_name;
        this->get_parameter("cloud_in_topic", cloud_topic_name);
        
        tree_.setResolution(resolution_);
        
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        auto qos_settings = rclcpp::SystemDefaultsQoS(); 
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_topic_name,
            qos_settings, 
            std::bind(&OctomapBuilder::cloudCallback, this, std::placeholders::_1));
        
        octomap_pub_ = this->create_publisher<octomap_msgs::msg::Octomap>("octomap_full", qos_settings);
        binary_map_pub_ = this->create_publisher<octomap_msgs::msg::Octomap>("octomap_binary", qos_settings);
        filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_cloud", qos_settings);
        
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&OctomapBuilder::reconfigureCallback, this, std::placeholders::_1));
        
        RCLCPP_INFO(this->get_logger(), "OctomapBuilder initialized.");
        RCLCPP_INFO(this->get_logger(), "Subscribing to PointCloud2 topic: '%s'", cloud_topic_name.c_str());
        RCLCPP_INFO(this->get_logger(), "Building map in frame: '%s'", world_frame_.c_str());
    }

private:
    rcl_interfaces::msg::SetParametersResult reconfigureCallback(const std::vector<rclcpp::Parameter> &parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        for (const auto &param : parameters) {
            if (param.get_name() == "resolution") {
                resolution_ = param.as_double();
                tree_.setResolution(resolution_);
            } else if (param.get_name() == "max_range") {
                max_range_ = param.as_double();
            } else if (param.get_name() == "min_range") {
                min_range_ = param.as_double();
            } else if (param.get_name() == "voxel_filter_size") {
                voxel_size_ = param.as_double();
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Reconfigured: res=%.3f, max_range=%.1f, voxel=%.3f", 
                    resolution_, max_range_, voxel_size_);
        return result;
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
        try {
            if (cloud_msg->header.frame_id.empty()) {
                RCLCPP_WARN(this->get_logger(), "PointCloud has empty frame_id! Skipping.");
                return;
            }
    
            geometry_msgs::msg::TransformStamped transform;
            
            transform = tf_buffer_->lookupTransform(
                world_frame_, cloud_msg->header.frame_id,
                tf2::TimePointZero,
                tf2::durationFromSec(1.0)
            );
    
            sensor_msgs::msg::PointCloud2 transformed_cloud;
            tf2::doTransform(*cloud_msg, transformed_cloud, transform);
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(transformed_cloud, *pcl_cloud);
            
            if (pcl_cloud->empty()) {
                RCLCPP_WARN(this->get_logger(), "Cloud is empty after transform/conversion. Skipping.");
                return;
            }
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_nan(new pcl::PointCloud<pcl::PointXYZ>);
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*pcl_cloud, *cloud_no_nan, indices);
            
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            voxel_filter.setInputCloud(cloud_no_nan);
            voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            voxel_filter.filter(*filtered_cloud);

            if (filtered_cloud->empty()) {
                RCLCPP_WARN(this->get_logger(), "Cloud is empty after filtering. Skipping.");
                return;
            }

            sensor_msgs::msg::PointCloud2 filtered_msg;
            pcl::toROSMsg(*filtered_cloud, filtered_msg);
            filtered_msg.header.stamp = this->get_clock()->now();
            filtered_msg.header.frame_id = world_frame_;
            filtered_cloud_pub_->publish(filtered_msg);

            octomap::point3d sensor_origin(
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            );

            octomap::Pointcloud octomap_cloud;
            for (const auto& point : *filtered_cloud) {
                octomap_cloud.push_back(point.x, point.y, point.z);
            }
            
            tree_.insertPointCloud(octomap_cloud, sensor_origin, max_range_);
            tree_.updateInnerOccupancy();

            publishMaps();
            
        } catch (const tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "Could not transform '%s' to '%s': %s",
                         cloud_msg->header.frame_id.c_str(), world_frame_.c_str(), ex.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "An unexpected error occurred in cloudCallback: %s", e.what());
        }
    }

    void publishMaps() {
        octomap_msgs::msg::Octomap full_map_msg;
        full_map_msg.header.frame_id = world_frame_;
        full_map_msg.header.stamp = this->get_clock()->now();
        if (octomap_msgs::fullMapToMsg(tree_, full_map_msg)) {
            octomap_pub_->publish(full_map_msg);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Error serializing full OctoMap");
        }
        
        octomap_msgs::msg::Octomap binary_map_msg;
        binary_map_msg.header.frame_id = world_frame_;
        binary_map_msg.header.stamp = this->get_clock()->now();
        binary_map_msg.binary = true;
        if (octomap_msgs::binaryMapToMsg(tree_, binary_map_msg)) {
            binary_map_pub_->publish(binary_map_msg);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Error serializing binary OctoMap");
        }
    }

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_pub_;
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr binary_map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    
    octomap::OcTree tree_;
    
    std::string world_frame_;
    double resolution_;
    double max_range_;
    double min_range_;
    double voxel_size_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);

    auto node = std::make_shared<OctomapBuilder>(options);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
