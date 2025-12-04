#ifndef MR_HET_COORD_ROS_VRP_NODE_HPP
#define MR_HET_COORD_ROS_VRP_NODE_HPP

#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include "mr_het_coord/map/map.hpp"
#include "mr_het_coord/planners/base_planner.hpp"
#include "mr_het_coord/planners/graph/prm.hpp"
#include "mr_het_coord/planners/map/rrt_star.hpp"
#include "mr_het_coord/utils/csv_writer.hpp"
#include "mr_het_coord/vrpsolver.hpp"
#include "mr_het_coord_ros/msg/vrp_solution.hpp"
#include "mr_het_coord_ros/utils.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "visualization_msgs/msg/marker_array.hpp"

struct ResultExp {
  int64_t exp_id;
  std::string method;
  std::string map;
  int rnd_seed;
  std::vector<float> route_total_cost;
  std::vector<float> route_cost_distance;
  std::vector<float> route_cost_time;
  std::vector<float> route_cost_traversability;
  std::vector<float> route_cost_collision;
  std::vector<float> route_cost_safety;
};

class MRVRPNode : public rclcpp::Node {
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
  std::vector<std::string> agent_types_str_;
  std::vector<AgentType> agent_types_;
  float step_size_;
  std::string map_name_;
  float cost_scaling_;
  float lambda_good_trav_;
  float lambda_bad_trav_;
  float gamma_collision_;
  float d_05_collision_;
  int solver_seconds_;
  bool fixed_random_seed_;
  int rnd_seed_;
  std::mt19937 rng_;
  int64_t experiment_id_;

  std::string paths_filename_;
  std::string results_path_;
  std::string tasks_filename_;
  std::vector<double> depot_position_;
  std::shared_ptr<GridMap> map_;
  std::string global_frame_;
  std::vector<std::vector<float>> tasks_;
  std::shared_ptr<BasePlanner> path_planner_;
  std::vector<std::vector<std::vector<float>>> path_, hom_path_, het_path_;
  std::vector<std::vector<std::vector<float>>> astar_tree_;

  void readROSParameters();

  std::vector<std::vector<float>> readTasksFromFile(std::string const &filename) const;

  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);

  // visualization

  visualization_msgs::msg::MarkerArray createTasksMarkers(
      std::vector<std::vector<float>> const &tasks, float const &scale) const;

  visualization_msgs::msg::MarkerArray createPathsMarkers(
      std::vector<std::vector<std::vector<float>>> const &paths,
      float const &scale, std::string const &ns,
      const float &height = 0.1f) const;

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

  void readPaths(
      std::string const &filename,
      std::vector<std::vector<std::vector<std::vector<float>>>> &paths);

  void solveVRP(VRPSolver &solver,
                std::vector<std::vector<int64_t>> distance_matrix,
                std::vector<std::vector<std::vector<std::vector<float>>>> paths,
                bool is_use_het, bool publish);

  void logResultsCSV(const ResultExp &result) const;
};

#endif