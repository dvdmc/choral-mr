#ifndef MR_HET_COORD_ROS_UTILS_HPP
#define MR_HET_COORD_ROS_UTILS_HPP

#include <nlohmann/json.hpp>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

#include "mr_het_coord/map/map.hpp"

nav_msgs::msg::OccupancyGrid createMapMsg(GridMap const& map, rclcpp::Time const& time) {
    nav_msgs::msg::OccupancyGrid msg;
    msg.header.frame_id = "map";
    msg.header.stamp = time;
    msg.info.resolution = map.resolution;
    msg.info.width = map.width_px;
    msg.info.height = map.height_px;
    msg.info.origin.position.x = map.origin[0];
    msg.info.origin.position.y = map.origin[1];
    msg.info.origin.position.z = 0.0;
    msg.info.origin.orientation.x = 0.0;
    msg.info.origin.orientation.y = 0.0;
    msg.info.origin.orientation.z = 0.0;
    for (float r = 0; r < map.height_px; r++) {
      for (float c = 0; c < map.width_px; c++) {
        msg.data.push_back(int(map.grid_map[r][c]));
      }
    }
    return msg;
}

using json = nlohmann::json;

// Paths are used to store the results from the path planning so they don't
// have to be recomputed for large environments
struct Paths
{
    std::vector<std::vector<std::vector<std::vector<float>>>> paths;
};

void to_json(json &j, Paths const &p)
{
    j["paths"] = p.paths;
}

void from_json(json const &j, Paths &p)
{
    j.at("paths").get_to(p.paths);
}

// The plan for the agents is saved in case that we want to use
// a trajectory generation outside of ROS
struct Plan
{
    int num_vehicles;
    std::vector<std::vector<std::vector<float>>> path;
};

void to_json(json &j, Plan const &p)
{
    j["num_vehicles"] = p.num_vehicles;
    for (size_t k = 0; k < p.path.size(); ++k)
    {
        j["agent" + std::to_string(k + 1)] = p.path[k];
    }
}

void from_json(json const &j, Plan &p)
{
    j.at("num_vehicles").get_to(p.num_vehicles);
    p.path.clear();
    p.path.resize(j.size());
    for (size_t k = 0; k < j.size(); ++k)
    {
        j.at("agent" + std::to_string(k + 1)).get_to(p.path[k]);
    }
}

#endif