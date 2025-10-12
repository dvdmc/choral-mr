#ifndef MR_HET_COORD_ROS_VRP_NODE_HPP
#define MR_HET_COORD_ROS_VRP_NODE_HPP

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2/LinearMath/Quaternion.h"

#include "mr_het_coord_ros/msg/vrp_solution.hpp"
#include "mr_het_coord_ros/utils.hpp"

#include "mr_het_coord/map/map.hpp"
#include "mr_het_coord/planners/base_planner.hpp"
#include "mr_het_coord/planners/graph/prm.hpp"
#include "mr_het_coord/planners/map/rrt_star.hpp"
#include "mr_het_coord/vrpsolver.hpp"

struct ResultExp
{
    std::string map;
    std::string method;
    std::vector<float> route_total_cost;
    std::vector<float> route_cost_distance;
    std::vector<float> route_cost_time;
    std::vector<float> route_cost_traversability;
    std::vector<float> route_cost_safety;
    int64_t rnd_seed;
};

class MRVRPNode : public rclcpp::Node
{
public:
    MRVRPNode();
    ~MRVRPNode();

private:
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_tasks_;
    rclcpp::Publisher<mr_het_coord_ros::msg::VrpSolution>::SharedPtr pub_vrp_sol_;

    rclcpp::TimerBase::SharedPtr pub_timer_;

    int num_vehicles_;
    std::vector<float> velocities_;
    std::vector<AgentType> agent_types_;
    float step_size_;
    std::string map_name_;
    float cost_scaling_;
    float lambda_good_trav_;
    float lambda_bad_trav_;
    float gamma_collision_;
    float d_05_collision_;
    int solver_seconds_;

    std::string paths_filename_;
    std::string results_path_;
    std::shared_ptr<GridMap> map_;
    std::vector<std::vector<float>> tasks_;
    std::shared_ptr<BasePlanner> path_planner_;
    std::vector<std::vector<std::vector<float>>> path_, hom_path_, het_path_;
    std::vector<std::vector<std::vector<float>>> astar_tree_;

    void readROSParameters();

    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    
    // visualization

    visualization_msgs::msg::MarkerArray createTasksMarkers(
        std::vector<std::vector<float>> const &tasks, float const &scale) const;

    visualization_msgs::msg::MarkerArray createPathsMarkers(
        std::vector<std::vector<std::vector<float>>> const &paths,
        float const &scale, std::string const &ns, const float &height = 0.1f) const;

    visualization_msgs::msg::MarkerArray createTreeMarkers(
        std::vector<std::vector<std::vector<float>>> const &edges,
        float const &scale, std::string const &ns) const;

    void removeAllMarkers() const;
    void pubTimerCallback();
    
    void savePath(std::vector<std::vector<std::vector<float>>> const &positions,
              std::string const &filename) const;

    void savePaths(
        const std::vector<std::vector<std::vector<std::vector<float>>>> &paths,
        std::string const &filename) const;

    void readPaths(std::string const &filename,
        std::vector<std::vector<std::vector<std::vector<float>>>> &paths);

    void solveVRP(
        VRPSolver &solver, std::vector<std::vector<int64_t>> distance_matrix,
        std::vector<std::vector<std::vector<std::vector<float>>>> paths,
        bool is_use_het, bool publish);
};

#endif