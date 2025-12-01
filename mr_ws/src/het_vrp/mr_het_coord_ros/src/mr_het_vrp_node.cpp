#include "mr_het_coord_ros/mr_het_vrp_node.hpp"

MRVRPNode::MRVRPNode() : Node("mr_coord_node"), rnd_seed_(42) {

  readROSParameters();

  experiment_id_ = std::chrono::system_clock::now().time_since_epoch().count();

  if (!fixed_random_seed_) {
    rnd_seed_ = experiment_id_ % 1000;
  }
  RCLCPP_INFO(this->get_logger(), "Random seed: %d", rnd_seed_);

  // Set random seed
  rng_.seed(rnd_seed_);

  pub_tasks_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "tasks_markers", 100);

  pub_vrp_sol_ = this->create_publisher<mr_het_coord_ros::msg::VrpSolution>(
      "vrp_sol", 100);

  sub_map_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "grid_map_in", 100,
      std::bind(&MRVRPNode::mapCallback, this, std::placeholders::_1));
  while (map_ == nullptr) {
    RCLCPP_INFO(this->get_logger(), "Waiting for map to be published on %s",
                sub_map_->get_topic_name());
    rclcpp::spin_some(this->get_node_base_interface());
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  pub_timer_ = this->create_wall_timer(
      std::chrono::seconds(1), std::bind(&MRVRPNode::pubTimerCallback, this));

  // Get tasks
  if(tasks_filename_ != "none") {
    tasks_ = readTasksFromFile(tasks_filename_);
  } else {
    tasks_ = map_->getTasksFromGridMap();
  }
  RCLCPP_INFO(this->get_logger(), "Found %ld tasks", tasks_.size());
  if(tasks_.size() == 0) {
    RCLCPP_ERROR(this->get_logger(), "No tasks found, shutting down");
    rclcpp::shutdown();
  }
  RCLCPP_INFO(this->get_logger(), "Running PRM");
  path_planner_ = std::make_shared<PRM>(*map_, tasks_, step_size_, 5.0f, rng_);

  VRPSolver solver;
  std::vector<std::vector<int64_t>> distance_matrix;
  std::vector<std::vector<std::vector<std::vector<float>>>> paths;

  bool use_file_paths = false;
  if (paths_filename_ == "none" || !use_file_paths) {
    RCLCPP_INFO(this->get_logger(), "Generating paths");
    auto start = std::chrono::high_resolution_clock::now();
    solver.distanceFromPathPlanner(tasks_, *path_planner_, distance_matrix,
                                   paths);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    RCLCPP_WARN(this->get_logger(), "Time taken for computing paths: %ld ms",
                duration.count());

    // Save paths to file
    paths_filename_ = "/root/mr_het_ws/paths/" + map_name_ + ".json";
    RCLCPP_WARN(this->get_logger(), "Saving path to file: %s",
                paths_filename_.c_str());
    savePaths(paths, paths_filename_);
  } else {
    RCLCPP_WARN(this->get_logger(), "Opening path from file: %s",
                paths_filename_.c_str());
    readPaths(paths_filename_, paths);
    solver.distanceFromPaths(paths, *path_planner_, distance_matrix);
  }

  // Create a unique
  RCLCPP_INFO(this->get_logger(), "Running EXP: homVRP");
  solveVRP(solver, distance_matrix, paths, false, true);
  RCLCPP_INFO(this->get_logger(), "Running EXP: hetVRP");
  solveVRP(solver, distance_matrix, paths, true, true);
  RCLCPP_INFO(this->get_logger(), "Finished experiments");

  // Uncomment below for batched experiments
  rclcpp::shutdown();
}
MRVRPNode::~MRVRPNode() {}

void MRVRPNode::pubTimerCallback() {
  RCLCPP_INFO(this->get_logger(), "Publishing markers");
  removeAllMarkers();

  if (tasks_.size() > 0) {
    visualization_msgs::msg::MarkerArray markers =
        createTasksMarkers(tasks_, map_->width_m * 0.01);
    pub_tasks_->publish(markers);
  }
  if (hom_path_.size() > 0) {
    visualization_msgs::msg::MarkerArray markers =
        createPathsMarkers(hom_path_, map_->width_m * 0.002, "homPath");
    pub_tasks_->publish(markers);
  }
  if (het_path_.size() > 0) {
    visualization_msgs::msg::MarkerArray markers =
        createPathsMarkers(het_path_, map_->width_m * 0.002, "hetPath");
    pub_tasks_->publish(markers);
  }
  std::vector<std::vector<std::vector<float>>> edges =
      std::static_pointer_cast<PRM>(path_planner_)->getEdges();
  if (edges.size() > 0) {
    visualization_msgs::msg::MarkerArray markers =
        createTreeMarkers(edges, map_->width_m * 0.001, "PRM");
    pub_tasks_->publish(markers);
  }
}

void MRVRPNode::readROSParameters() {
  // Agent info
  this->declare_parameter("num_vehicles", 0);
  this->declare_parameter("velocities", std::vector<double>({}));
  this->declare_parameter("agent_types", std::vector<std::string>({}));

  // Mission info
  this->declare_parameter("step_size", 0.0);

  this->declare_parameter("map_name", "");
  this->declare_parameter("paths_filename", "none");
  this->declare_parameter("results_path", "none");

  this->declare_parameter("tasks_filename", "none");

  // VRP config
  this->declare_parameter("cost_scaling", 5.0f);
  this->declare_parameter("lambda_good_trav", 0.00001);
  this->declare_parameter("lambda_bad_trav", 5.0);
  this->declare_parameter("gamma_collision", 20.0);
  this->declare_parameter("d_05_collision", 0.1);
  this->declare_parameter("min_max_solver", "");

  this->declare_parameter("solver_seconds", 30);
  this->declare_parameter("fixed_random_seed", true);

  std::vector<double> temp;
  this->get_parameter("num_vehicles", num_vehicles_);
  this->get_parameter("velocities", temp);
  velocities_ = std::vector<float>(temp.begin(), temp.end());
  this->get_parameter("agent_types", agent_types_str_);

  this->get_parameter("step_size", step_size_);

  this->get_parameter("map_name", map_name_);
  this->get_parameter("paths_filename", paths_filename_);
  this->get_parameter("results_path", results_path_);

  this->get_parameter("tasks_filename", tasks_filename_);

  this->get_parameter("cost_scaling", cost_scaling_);
  this->get_parameter("lambda_good_trav", lambda_good_trav_);
  this->get_parameter("lambda_bad_trav", lambda_bad_trav_);
  this->get_parameter("gamma_collision", gamma_collision_);
  this->get_parameter("d_05_collision", d_05_collision_);
  this->get_parameter("solver_seconds", solver_seconds_);
  this->get_parameter("fixed_random_seed", fixed_random_seed_);

  for (auto s : agent_types_str_) {
    agent_types_.push_back(stringToAgentType(s));
  }
  RCLCPP_INFO(this->get_logger(),
              "------------------VRP Node Parameters-----------------");
  RCLCPP_INFO(this->get_logger(), "Num vehicles: %d", num_vehicles_);
  RCLCPP_INFO(this->get_logger(), "Velocities: ");
  for (auto v : velocities_) RCLCPP_INFO(this->get_logger(), "%.2f ", v);
  RCLCPP_INFO(this->get_logger(), "Agent types: ");
  for (auto t : agent_types_) RCLCPP_INFO(this->get_logger(), "%d ", int(t));
  RCLCPP_INFO(this->get_logger(), "Step size: %.2f", step_size_);
  RCLCPP_INFO(this->get_logger(), "Map name: %s", map_name_.c_str());
  RCLCPP_INFO(this->get_logger(), "Paths filename: %s",
              paths_filename_.c_str());
  RCLCPP_INFO(this->get_logger(), "Results path: %s", results_path_.c_str());
  RCLCPP_INFO(this->get_logger(), "Tasks filename: %s", tasks_filename_.c_str());
  RCLCPP_INFO(this->get_logger(), "Cost scaling: %.2f", cost_scaling_);
  RCLCPP_INFO(this->get_logger(), "Lambda good trav: %.2f", lambda_good_trav_);
  RCLCPP_INFO(this->get_logger(), "Lambda bad trav: %.2f", lambda_bad_trav_);
  RCLCPP_INFO(this->get_logger(), "Gamma collision: %.2f", gamma_collision_);
  RCLCPP_INFO(this->get_logger(), "D_05 collision: %.2f", d_05_collision_);
  RCLCPP_INFO(this->get_logger(), "Solver seconds: %d", solver_seconds_);
  RCLCPP_INFO(this->get_logger(),
              "-----------------------------------------------------");
}

void MRVRPNode::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  if (map_ != nullptr) {
    RCLCPP_WARN_ONCE(this->get_logger(),
                     "Map already received, ignoring new map");
    return;
  }
  RCLCPP_INFO(this->get_logger(), "Map received");
  float map_resolution = msg->info.resolution;
  auto map_center = std::vector<float>{float(msg->info.origin.position.x),
                                       float(msg->info.origin.position.y)};
  map_ = std::make_shared<GridMap>(map_resolution, map_center);
  // Build a matrix from the data
  std::vector<std::vector<int>> grid_map(msg->info.height,
                                         std::vector<int>(msg->info.width, 0));
  for (size_t i = 0; i < msg->data.size(); i++) {
    grid_map[i / msg->info.width][i % msg->info.width] = msg->data[i];
  }
  map_->setMap(msg->info.width, msg->info.height, grid_map);
}

visualization_msgs::msg::MarkerArray MRVRPNode::createTasksMarkers(
    std::vector<std::vector<float>> const &tasks, float const &scale) const {
  visualization_msgs::msg::MarkerArray msg;
  for (size_t i = 0; i < tasks.size(); ++i) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->now();
    marker.ns = "tasks";
    marker.id = i;
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::msg::Marker::MODIFY;
    marker.pose.position.x = tasks[i][0];
    marker.pose.position.y = tasks[i][1];
    marker.pose.position.z = 0.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = scale;
    marker.scale.y = scale;
    marker.scale.z = scale;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.text = std::to_string(i);
    msg.markers.push_back(marker);
  }
  return msg;
}

std::vector<std::vector<float>> MRVRPNode::readTasksFromFile(std::string const &filename) const
{
  // Tasks are a json file with a list of [x,y] positions
  // Example: {tasks: [[1.0,2.0],[3.0,4.0],...]}
  std::ifstream file;
  file.open(filename);
  if (!file.is_open()) {
    std::cout << "Couldn't open tasks file at " << filename << "!" << std::endl;
    rclcpp::shutdown();
    return {};
  }
  std::cout << "Reading tasks from file: " << filename << std::endl;
  json j;
  file >> j;
  std::vector<std::vector<float>> tasks = j["tasks"].get<std::vector<std::vector<float>>>();
  file.close();
  return tasks;
}

visualization_msgs::msg::MarkerArray MRVRPNode::createPathsMarkers(
    std::vector<std::vector<std::vector<float>>> const &paths,
    float const &scale, std::string const &ns, const float &height) const {
  std::vector<std::vector<float>> colors{
      {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
  visualization_msgs::msg::MarkerArray msg;
  for (size_t k = 0; k < paths.size(); ++k) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->now();
    marker.ns = ns;
    marker.id = k;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::MODIFY;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = scale;
    marker.scale.y = scale;
    marker.scale.z = scale;
    marker.color.a = 1.0;
    marker.color.r = colors[k][0];
    marker.color.g = colors[k][1];
    marker.color.b = colors[k][2];
    for (size_t i = 0; i < paths[k].size(); ++i) {
      geometry_msgs::msg::Point p;
      p.x = paths[k][i][0];
      p.y = paths[k][i][1];
      p.z = height;
      marker.points.push_back(p);
    }
    msg.markers.push_back(marker);
  }
  return msg;
}

visualization_msgs::msg::MarkerArray MRVRPNode::createTreeMarkers(
    std::vector<std::vector<std::vector<float>>> const &edges,
    float const &scale, std::string const &ns) const {
  visualization_msgs::msg::MarkerArray msg;
  for (size_t i = 0; i < edges.size(); ++i) {
    std::vector<std::vector<float>> edge = edges[i];
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->now();
    marker.ns = ns;
    marker.id = i;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::MODIFY;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = scale;
    marker.scale.y = scale;
    marker.scale.z = scale;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;

    // First point
    geometry_msgs::msg::Point p;
    p.x = edge[0][0];
    p.y = edge[0][1];
    p.z = 0.1;
    marker.points.push_back(p);

    // Second point
    p.x = edge[1][0];
    p.y = edge[1][1];
    p.z = 0.1;
    marker.points.push_back(p);

    msg.markers.push_back(marker);
  }
  return msg;
}

void MRVRPNode::removeAllMarkers() const {
  visualization_msgs::msg::MarkerArray msg;
  visualization_msgs::msg::Marker marker;
  marker.action = visualization_msgs::msg::Marker::DELETEALL;
  msg.markers.push_back(marker);
  pub_tasks_->publish(msg);
}

void MRVRPNode::savePath(
    std::vector<std::vector<std::vector<float>>> const &positions,
    std::string const &filename) const {
  std::ofstream file;
  file.open(filename);
  json j;

  Plan p;
  p.path = positions;
  p.num_vehicles = num_vehicles_;
  j = p;

  file << j.dump(4) << std::endl;
  file.close();
}

void MRVRPNode::savePaths(
    const std::vector<std::vector<std::vector<std::vector<float>>>> &paths,
    std::string const &filename) const {
  std::ofstream file;
  file.open(filename);
  json j;
  Paths p;
  p.paths = paths;
  j = p;

  file << j.dump(4) << std::endl;
  file.close();
}

void MRVRPNode::readPaths(
    std::string const &filename,
    std::vector<std::vector<std::vector<std::vector<float>>>> &paths) {
  std::ifstream file;
  file.open(filename);
  if (!file.is_open()) {
    std::cout << "Couldn't open file!" << std::endl;
    rclcpp::shutdown();
    return;
  }
  json j;
  file >> j;
  Paths p;
  j.get_to(p);
  paths = p.paths;
  file.close();
}

void MRVRPNode::solveVRP(
    VRPSolver &solver, std::vector<std::vector<int64_t>> distance_matrix,
    std::vector<std::vector<std::vector<std::vector<float>>>> paths,
    bool is_use_het, bool publish) {
  std::vector<std::vector<std::vector<int64_t>>> cost_matrices;
  std::vector<float> velocities;
  std::string method_str;
  path_.clear();
  if (!is_use_het) {
    hom_path_.clear();
    solver.uniformCostMatrices(distance_matrix, num_vehicles_, cost_matrices);
    // Use the same velocity for all vehicles
    for (int i = 0; i < num_vehicles_; i++) {
      velocities.push_back(1.0f);
    }
    method_str = "hom";
    std::cout << "VRP with hom:" << std::endl;
  } else {
    het_path_.clear();
    solver.hetPlatformCostMatrices(*map_, *path_planner_, num_vehicles_, paths,
                                   agent_types_, cost_matrices,
                                   lambda_good_trav_, lambda_bad_trav_,
                                   gamma_collision_, d_05_collision_);
    velocities = velocities_;
    method_str = "het";
    std::cout << "VRP with het:" << std::endl;
  }

  std::vector<std::vector<int64_t>> solution;
  std::vector<int64_t> resulting_route_cost;

  std::cout << "Max Span" << std::endl;
  solution = solver.solveWithMaxSpan(distance_matrix, cost_matrices,
                                     cost_scaling_, velocities, num_vehicles_,
                                     0, solver_seconds_, resulting_route_cost);

  // Since we already have the paths, use them
  for (size_t k = 0; k < solution.size(); ++k) {  // For each vehicle
    std::vector<std::vector<float>> agent_path;
    for (size_t i = 0; i < solution[k].size() - 1; ++i) {  // For each node
      int from_task = solution[k][i];
      int to_task = solution[k][i + 1];
      std::vector<std::vector<float>> path_segment = paths[from_task][to_task];
      for (size_t j = 0; j < path_segment.size() - 1; ++j) {
        agent_path.push_back(path_segment[j]);
      }
      // Add the last one
      if (i == solution[k].size() - 2) {
        agent_path.push_back(path_segment[path_segment.size() - 1]);
      }
    }
    path_.push_back(agent_path);
  }
  if (solution.empty()) {
    std::cout << "No solution found for VRP!" << std::endl;
    return;
  }
  std::cout << "Finish computing edge. Publishing and saving results..."
            << std::endl;

  if (!is_use_het) {
    hom_path_ = path_;
  } else {
    het_path_ = path_;
  }

  // OUT Store ResultsEx
  ResultExp result;
  result.exp_id = experiment_id_;
  result.method = "maxSpan_" + method_str;
  result.map = map_name_;
  result.rnd_seed = rnd_seed_;
  result.route_total_cost.resize(num_vehicles_);
  result.route_cost_distance.resize(num_vehicles_);
  result.route_cost_time.resize(num_vehicles_);
  result.route_cost_traversability.resize(num_vehicles_);
  result.route_cost_collision.resize(num_vehicles_);
  result.route_cost_safety.resize(num_vehicles_);
  for (int k = 0; k < num_vehicles_; ++k) {
    result.route_total_cost[k] = resulting_route_cost[k];
    result.route_cost_distance[k] = path_planner_->computeDistance(path_[k]);
    result.route_cost_time[k] = result.route_cost_distance[k] / velocities_[k];
    result.route_cost_traversability[k] =
        path_planner_->computeTraversabilityCost(
            *map_, path_[k], agent_types_[k], lambda_good_trav_,
            lambda_bad_trav_);
    result.route_cost_collision[k] = path_planner_->computeCollisionCost(
        *map_, path_[k], agent_types_[k], gamma_collision_, d_05_collision_);
    result.route_cost_safety[k] = path_planner_->computeSafetyCost(
        *map_, path_[k], agent_types_[k], lambda_good_trav_, lambda_bad_trav_,
        gamma_collision_, d_05_collision_);
  }

  // Save results as a CSV file
  logResultsCSV(result);

  std::cout << "Finished exp, writing paths" << std::endl;

  std::string filename = results_path_ + "plan_" + method_str + ".json";
  savePath(path_, filename);

  // Send solution msg
  std::cout << "Finished exp, publishing" << std::endl;

  if (publish) {
    auto msg = mr_het_coord_ros::msg::VrpSolution();
    msg.header.stamp = this->now();
    msg.header.frame_id = "map";
    msg.ns = method_str + "VRP";
    msg.num_agents = num_vehicles_;
    for (int k = 0; k < num_vehicles_; ++k) {
      geometry_msgs::msg::PoseArray poses;
      // Skip initial pose
      for (size_t i = 1; i < path_[k].size(); ++i) {
        geometry_msgs::msg::Pose pose;
        pose.position.x = path_[k][i][0];
        pose.position.y = path_[k][i][1];
        if (agent_types_[k] == AgentType::GROUND) {
          pose.position.z = 0.0;
        } else if (agent_types_[k] == AgentType::AERIAL) {
          pose.position.z = 1.0;
        }
        // Orient towards next goal
        if (i < path_[k].size() - 1) {
          float dx = path_[k][i + 1][0] - path_[k][i][0];
          float dy = path_[k][i + 1][1] - path_[k][i][1];
          float angle = std::atan2(dy, dx);
          tf2::Quaternion q(0.0, 0.0, 0.0, angle);
          pose.orientation.x = q.x();
          pose.orientation.y = q.y();
          pose.orientation.z = q.z();
          pose.orientation.w = q.w();
        } else {
          float dx = path_[k][i][0] - path_[k][i - 1][0];
          float dy = path_[k][i][1] - path_[k][i - 1][1];
          float angle = std::atan2(dy, dx);
          tf2::Quaternion q(0.0, 0.0, 0.0, angle);
          pose.orientation.x = q.x();
          pose.orientation.y = q.y();
          pose.orientation.z = q.z();
          pose.orientation.w = q.w();
        }

        poses.poses.push_back(pose);
      }
      msg.paths.push_back(poses);
    }

    pub_vrp_sol_->publish(msg);
  }
}

void MRVRPNode::logResultsCSV(const ResultExp &result) const {
  // Prepare CSV path
  std::string folder = results_path_;
  std::filesystem::create_directories(folder);
  std::string filename = folder + "results.csv";

  // Header in long format
  std::vector<std::string> header = {"exp_id",
                                     "method",
                                     "map",
                                     "rnd_seed",
                                     "agent_id",
                                     "agent_type",
                                     "velocity",
                                     "total_cost",
                                     "distance",
                                     "time",
                                     "traversability",
                                     "collision",
                                     "safety",
                                     "step_size",
                                     "cost_scaling",
                                     "lambda_good_trav",
                                     "lambda_bad_trav",
                                     "gamma_collision",
                                     "d_05_collision",
                                     "solver_seconds"};

  // Initialize writer (will only write header if file does not exist)
  CSVWriter writer(filename, header);

  // Loop over agents
  for (int k = 0; k < num_vehicles_; ++k) {
    writer.writeRow(
        // Experiment metadata
        result.exp_id, result.method, result.map, result.rnd_seed,

        // Agent info
        k + 1, // agent_id
        agent_types_str_[k],
        velocities_[k],

        // Performance metrics
        result.route_total_cost[k], result.route_cost_distance[k],
        result.route_cost_time[k], result.route_cost_traversability[k],
        result.route_cost_collision[k], result.route_cost_safety[k],

        // Solver parameters
        step_size_, cost_scaling_, lambda_good_trav_, lambda_bad_trav_,
        gamma_collision_, d_05_collision_, solver_seconds_);
  }

  RCLCPP_INFO(this->get_logger(), "Logged results for experiment %ld to %s",
              result.exp_id, filename.c_str());
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MRVRPNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
