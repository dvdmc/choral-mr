#ifndef MR_HET_COORD_ROS_MAP_NODE_HPP
#define MR_HET_COORD_ROS_MAP_NODE_HPP

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

#include "mr_het_coord_ros/utils.hpp"

#include "mr_het_coord/map/map.hpp"

class MRMapNode : public rclcpp::Node
{
public:
  MRMapNode();
  ~MRMapNode();

private:
    // This is simply for visualization in RViZ
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_map_;

    // Timer for visualization
    rclcpp::TimerBase::SharedPtr pub_timer_;

    // For loading the map from a file
    bool load_map_from_file_;
    bool pub_updates_only_;
    bool updated_;
    std::string map_filename_;
    std::string map_name_;
    float map_resolution_;
    std::vector<float> map_center_;

    // For receiving the map as a topic
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;

    std::shared_ptr<GridMap> map_;

    void readROSParameters();

    void pubTimerCallback();

    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
};

#endif