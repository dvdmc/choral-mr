#include "bloomxai_ros/bloomxai_server.hpp"

#include <pcl/filters/radius_outlier_removal.h>

#include "bloomxai_ros/semantic_utils.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

namespace {
template <typename T>
bool update_param(const std::vector<rclcpp::Parameter>& p, const std::string& name, T& value) {
  auto it = std::find_if(p.cbegin(), p.cend(), [&name](const rclcpp::Parameter& parameter) {
    return parameter.get_name() == name;
  });
  if (it != p.cend()) {
    value = it->template get_value<T>();
    return true;
  }
  return false;
}
}  // namespace

namespace bloomxai_server {
BloomxaiServer::BloomxaiServer(const rclcpp::NodeOptions& node_options)
    : Node("bloomxai_server_node", node_options) {
  using std::placeholders::_1;
  using std::placeholders::_2;

  {
    world_frame_id_ = declare_parameter("frame_id", "map");
    base_frame_id_ = declare_parameter(
        "base_frame_id", "base_footprint");  // NOTE: Not used since we get it from PCD message
  }

  {
    rcl_interfaces::msg::ParameterDescriptor occupancy_min_z_desc;
    occupancy_min_z_desc.description =
        "Minimum height of occupied cells to consider in the final map";
    rcl_interfaces::msg::FloatingPointRange occupancy_min_z_range;
    occupancy_min_z_range.from_value = -100.0;
    occupancy_min_z_range.to_value = 100.0;
    occupancy_min_z_desc.floating_point_range.push_back(occupancy_min_z_range);
    occupancy_min_z_ = declare_parameter("occupancy_min_z", -4.0, occupancy_min_z_desc);
  }
  {
    rcl_interfaces::msg::ParameterDescriptor occupancy_max_z_desc;
    occupancy_max_z_desc.description =
        "Maximum height of occupied cells to consider in the final map";
    rcl_interfaces::msg::FloatingPointRange occupancy_max_z_range;
    occupancy_max_z_range.from_value = -100.0;
    occupancy_max_z_range.to_value = 100.0;
    occupancy_max_z_desc.floating_point_range.push_back(occupancy_max_z_range);
    occupancy_max_z_ = declare_parameter("occupancy_max_z", 15.0, occupancy_max_z_desc);
  }

  {
    rcl_interfaces::msg::ParameterDescriptor max_range_desc, min_range_desc;
    max_range_desc.description = "Sensor maximum range";
    min_range_desc.description = "Sensor minimum range";
    rcl_interfaces::msg::FloatingPointRange max_range_range, min_range_range;
    max_range_range.from_value = -1.0;
    max_range_range.to_value = 100.0;
    min_range_range.from_value = -1.0;
    min_range_range.to_value = 10.0;
    max_range_desc.floating_point_range.push_back(max_range_range);
    min_range_desc.floating_point_range.push_back(min_range_range);
    max_range_ = declare_parameter("sensor_model.max_range", -1.0, max_range_desc);
    min_range_ = declare_parameter("sensor_model.min_range", -1.0, min_range_desc);
  }

  {
    rcl_interfaces::msg::ParameterDescriptor sem_dim_desc;
    sem_dim_desc.description = "Semantic dimension";
    rcl_interfaces::msg::IntegerRange sem_dim_range;
    sem_dim_range.from_value = 1;
    sem_dim_range.to_value = 1200;
    sem_dim_desc.integer_range.push_back(sem_dim_range);
    sem_dim_ = declare_parameter("sem_dim", 100, sem_dim_desc);
    initial_sem_val_ = 1.0f / sem_dim_;
    label_to_rgb_ = getLabelMap(sem_dim_);
    sim_to_rgb_ = [](float sim) { return jetColor(sim); };
  }

  res_ = declare_parameter("resolution", 0.1);

  rcl_interfaces::msg::ParameterDescriptor prob_hit_desc;
  prob_hit_desc.description =
      "Probabilities for hits in the sensor model when dynamically building a map";
  rcl_interfaces::msg::FloatingPointRange prob_hit_range;
  prob_hit_range.from_value = 0.5;
  prob_hit_range.to_value = 1.0;
  prob_hit_desc.floating_point_range.push_back(prob_hit_range);
  const double prob_hit = declare_parameter("sensor_model.hit", 0.7, prob_hit_desc);

  rcl_interfaces::msg::ParameterDescriptor prob_miss_desc;
  prob_miss_desc.description =
      "Probabilities for misses in the sensor model when dynamically building a map";
  rcl_interfaces::msg::FloatingPointRange prob_miss_range;
  prob_miss_range.from_value = 0.0;
  prob_miss_range.to_value = 0.5;
  prob_miss_desc.floating_point_range.push_back(prob_miss_range);
  const double prob_miss = declare_parameter("sensor_model.miss", 0.4, prob_miss_desc);

  rcl_interfaces::msg::ParameterDescriptor prob_min_desc;
  prob_min_desc.description = "Minimum probability for clamping when dynamically building a map";
  rcl_interfaces::msg::FloatingPointRange prob_min_range;
  prob_min_range.from_value = 0.0;
  prob_min_range.to_value = 1.0;
  prob_min_desc.floating_point_range.push_back(prob_min_range);
  const double thres_min = declare_parameter("sensor_model.min", 0.12, prob_min_desc);

  rcl_interfaces::msg::ParameterDescriptor prob_max_desc;
  prob_max_desc.description = "Maximum probability for clamping when dynamically building a map";
  rcl_interfaces::msg::FloatingPointRange prob_max_range;
  prob_max_range.from_value = 0.0;
  prob_max_range.to_value = 1.0;
  prob_max_desc.floating_point_range.push_back(prob_max_range);
  const double thres_max = declare_parameter("sensor_model.max", 0.97, prob_max_desc);

  // Semantic configs. Add for probabilities and features
  rcl_interfaces::msg::ParameterDescriptor semantic_type_desc;
  semantic_type_desc.description = "Semantic type used in the map: probabilities or features";
  const std::string semantic_type_str =
      declare_parameter("semantics.semantic_type", "probabilities", semantic_type_desc);
  // Transform string to enum
  semantic_type_ = semantic_type_map.at(semantic_type_str);

  // Declare the set of problematic and task classes
  std::vector<int64_t> tmp;
  this->declare_parameter("semantics.problematic_classes", std::vector<int64_t>{1});
  if(!this->get_parameter("semantics.problematic_classes", tmp)){
    RCLCPP_ERROR(this->get_logger(), "Problematic classes not configured, using default!");
  }
  problematic_classes_.assign(tmp.begin(), tmp.end());
  this->declare_parameter("semantics.task_classes", std::vector<int64_t>{2});
  if(!this->get_parameter("semantics.task_classes", tmp)){
    RCLCPP_ERROR(this->get_logger(), "Task classes not configured, using default!");
  }
  task_classes_.assign(tmp.begin(), tmp.end());

  // Declar the rest of paramters
  rcl_interfaces::msg::ParameterDescriptor alpha_reg_desc;
  alpha_reg_desc.description = "Alpha regularization parameter";
  rcl_interfaces::msg::FloatingPointRange alpha_reg_range;
  alpha_reg_range.from_value = 0.0;
  alpha_reg_range.to_value = 1.0;
  alpha_reg_desc.floating_point_range.push_back(alpha_reg_range);
  const double alpha_reg = declare_parameter("semantics.alpha_reg", 0.3, alpha_reg_desc);

  rcl_interfaces::msg::ParameterDescriptor sem_prob_min_desc;
  sem_prob_min_desc.description =
      "Minimum probability for clamping when dynamically building a map";
  rcl_interfaces::msg::FloatingPointRange sem_prob_min_range;
  sem_prob_min_range.from_value = 0.0;
  sem_prob_min_range.to_value = 1.0;
  sem_prob_min_desc.floating_point_range.push_back(sem_prob_min_range);
  const double sem_thres_min = declare_parameter("semantics.min", 0.12, sem_prob_min_desc);

  rcl_interfaces::msg::ParameterDescriptor sem_prob_max_desc;
  sem_prob_max_desc.description =
      "Maximum probability for clamping when dynamically building a map";
  rcl_interfaces::msg::FloatingPointRange sem_prob_max_range;
  sem_prob_max_range.from_value = 0.0;
  sem_prob_max_range.to_value = 1.0;
  sem_prob_max_desc.floating_point_range.push_back(sem_prob_max_range);
  const double sem_thres_max = declare_parameter("semantics.max", 0.97, sem_prob_max_desc);

  rcl_interfaces::msg::ParameterDescriptor sem_occ_th_desc;
  sem_occ_th_desc.description = "Threshold for deleting semantics when dynamically building a map";
  rcl_interfaces::msg::FloatingPointRange sem_prob_th_range;
  sem_prob_th_range.from_value = 0.0;
  sem_prob_th_range.to_value = 1.0;
  sem_occ_th_desc.floating_point_range.push_back(sem_prob_th_range);
  const double sem_occ_thres = declare_parameter("semantics.occ_thres", 0.5, sem_occ_th_desc);

  // Add an index for similarity
  rcl_interfaces::msg::ParameterDescriptor sim_index_desc;
  sim_index_desc.description = "Index for similarity";
  const int sim_index = declare_parameter("semantics.sim_query_index", 1, sim_index_desc);
  sim_query_index_ = sim_index;
  // The queries are modified through a service since it will come from a Python node

  // initialize bloomxai object & params
  RCLCPP_INFO(get_logger(), "Voxel resolution %f", res_);

  // Dump config
  RCLCPP_INFO(get_logger(), "world_frame_id %s", world_frame_id_.c_str());
  RCLCPP_INFO(get_logger(), "base_frame_id %s", base_frame_id_.c_str());
  RCLCPP_INFO(get_logger(), "occupancy_min_z %f", occupancy_min_z_);
  RCLCPP_INFO(get_logger(), "occupancy_max_z %f", occupancy_max_z_);
  RCLCPP_INFO(get_logger(), "min_range %f", min_range_);
  RCLCPP_INFO(get_logger(), "max_range %f", max_range_);
  RCLCPP_INFO(get_logger(), "sem_dim %d", sem_dim_);
  RCLCPP_INFO(get_logger(), "initial_sem_val %f", initial_sem_val_);
  RCLCPP_INFO(get_logger(), "prob_hit %f", prob_hit);
  RCLCPP_INFO(get_logger(), "prob_miss %f", prob_miss);
  RCLCPP_INFO(get_logger(), "thres_min %f", thres_min);
  RCLCPP_INFO(get_logger(), "thres_max %f", thres_max);

  RCLCPP_INFO(get_logger(), "Semantic type: %s", semantic_type_str.c_str());

  RCLCPP_INFO(get_logger(), "Problematic classes:");
  for (const int64_t& c : problematic_classes_) {
    RCLCPP_INFO(get_logger(), "%d", c);
  }

  RCLCPP_INFO(get_logger(), "Task classes:");
  for (const int64_t& c : task_classes_) {
    RCLCPP_INFO(get_logger(), "%d", c);
  }
  if (semantic_type_ == SemanticType::PROBABILITIES) {
    RCLCPP_INFO(get_logger(), "alpha_reg %f", alpha_reg);
    RCLCPP_INFO(get_logger(), "sem_thres_min %f", sem_thres_min);
    RCLCPP_INFO(get_logger(), "sem_thres_max %f", sem_thres_max);
  } else if (semantic_type_ == SemanticType::FEATURES) {
    RCLCPP_INFO(get_logger(), "sem_thres %f", sem_occ_thres);
    RCLCPP_INFO(get_logger(), "sim_index %d", sim_index);
  }

  // Create semantic operator
  std::unique_ptr<Bloomxai::BaseSemanticOperator> semantic_operator;
  if (semantic_type_ == SemanticType::PROBABILITIES) {
    semantic_operator = std::make_unique<Bloomxai::ProbabilitySemanticOperator>(sem_dim_);

    Bloomxai::ProbabilitySemanticOperator::ProbOptions prob_options;
    prob_options.alpha_reg = alpha_reg;
    prob_options.clamp_min_prob = bloomxai_->logods(sem_thres_min);
    prob_options.clamp_max_prob = bloomxai_->logods(sem_thres_max);
    semantic_operator->setOptions(prob_options);

  } else if (semantic_type_ == SemanticType::FEATURES) {
    semantic_operator = std::make_unique<Bloomxai::FeatureSemanticOperator>(sem_dim_);

    Bloomxai::FeatureSemanticOperator::FeatOptions feat_options;
    feat_options.occ_thres_logods = bloomxai_->logods(sem_occ_thres);
    semantic_operator->setOptions(feat_options);
  }

  bloomxai_ = std::make_unique<BloomxaiT>(res_, sem_dim_, std::move(semantic_operator));

  BloomxaiT::Options options;
  options.prob_miss_log = bloomxai_->logods(prob_miss);
  options.prob_hit_log = bloomxai_->logods(prob_hit);
  options.clamp_min_log = bloomxai_->logods(thres_min);
  options.clamp_max_log = bloomxai_->logods(thres_max);

  bloomxai_->setOptions(options);

  latched_topics_ = declare_parameter("latch", false);
  if (latched_topics_) {
    RCLCPP_INFO(
        get_logger(),
        "Publishing latched (single publish will take longer, "
        "all topics are prepared)");
  } else {
    RCLCPP_INFO(
        get_logger(),
        "Publishing non-latched (topics are only prepared as needed, "
        "will only be re-published on map change");
  }

  auto qos = latched_topics_ ? rclcpp::QoS{1}.transient_local() : rclcpp::QoS{1};
  marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("bloomxai/marker", qos);
  grid_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("grid_map", qos);

  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  using std::chrono_literals::operator""s;

  point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "cloud_in", rclcpp::SensorDataQoS(),
      std::bind(&BloomxaiServer::insertCloudCallback, this, std::placeholders::_1));

  save_srv_ = create_service<bloomxai_ros::srv::SaveMap>(
      "~/save_map",
      std::bind(
          &BloomxaiServer::saveMapCallback, this, std::placeholders::_1, std::placeholders::_2));

  save_pgm_srv_ = create_service<bloomxai_ros::srv::SaveMap>(
      "~/save_pgm_map",
      std::bind(
          &BloomxaiServer::savePGMMapCallback, this, std::placeholders::_1, std::placeholders::_2));

  load_srv_ = create_service<bloomxai_ros::srv::LoadMap>(
      "~/load_map",
      std::bind(
          &BloomxaiServer::loadMapCallback, this, std::placeholders::_1, std::placeholders::_2));

  if (semantic_type_ == SemanticType::FEATURES) {
    set_queries_srv_ = create_service<sensors_tools_msgs::srv::SetQueries>(
        "~/set_queries", std::bind(
                             &BloomxaiServer::setQueriesCallback, this, std::placeholders::_1,
                             std::placeholders::_2));
  }

  reset_srv_ =
      create_service<ResetSrv>("~/reset", std::bind(&BloomxaiServer::resetSrv, this, _1, _2));

  // set parameter callback
  set_param_res_ =
      this->add_on_set_parameters_callback(std::bind(&BloomxaiServer::onParameter, this, _1));

  // create timer for publishing visualization
  vis_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(1000), std::bind(&BloomxaiServer::publishAll, this));
}

bool BloomxaiServer::setQueriesCallback(
    const std::shared_ptr<sensors_tools_msgs::srv::SetQueries::Request> request,
    const std::shared_ptr<sensors_tools_msgs::srv::SetQueries::Response> response) {
  int num_queries = request->num_queries;
  std::vector<float> flat_embeddings = request->flat_embeddings;

  Bloomxai::FeatureSemanticOperator::FeatOptions options;
  // Read param
  double occ_thres = get_parameter("semantics.occ_thres").as_double();
  options.occ_thres_logods = bloomxai_->logods(occ_thres);
  options.num_queries = num_queries;
  if (num_queries * sem_dim_ != flat_embeddings.size()) {
    response->success = false;
    response->message = "flat_embeddings dimension mismatch";
    return false;
  }
  Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> temp_map(
      flat_embeddings.data(), num_queries, sem_dim_);

  options.query_embeddings.resize(num_queries, sem_dim_);
  options.query_embeddings = temp_map;
  bloomxai_->semantic_operator->setOptions(options);
  response->success = true;

  response->message = "Queries set successfully.";
  RCLCPP_INFO(get_logger(), "Set %d text queries", num_queries);
  return true;
}

bool BloomxaiServer::saveMapCallback(
    const std::shared_ptr<bloomxai_ros::srv::SaveMap::Request> request,
    const std::shared_ptr<bloomxai_ros::srv::SaveMap::Response> response) {
  try {
    bloomxai_->serializeToFile(request->filename);
    RCLCPP_INFO(get_logger(), "Map saved to %s", request->filename.c_str());
    response->success = true;
    response->message = "Map saved successfully.";
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Failed to save map: %s", e.what());
    response->success = false;
    response->message = e.what();
    return false;
  }
}

bool BloomxaiServer::sendGridMap() const {
  std::vector<float> min, max;
  std::vector<int> map_size;
  std::vector<std::vector<int>> matrix;
  
  double th_z = 0.8;

  {
    std::lock_guard<std::mutex> lock(bloomxai_mutex_);
    if (!bloomxai_->semantic_operator->isReady()) {
      return false;
    }
    bloomxai_->getMapLimits(min, max);
    map_size = bloomxai_->getMapXYSize();
    std::cout << "Map size: " << map_size[0] << ", " << map_size[1] << std::endl;
    for (int i = 0; i < 3; i++) {
      max[i] -= bloomxai_->getResolution() / 2.0;
    }

    matrix = bloomxai_->generate2DGridMap(min, max, map_size, problematic_classes_, task_classes_, th_z);
  }

  nav_msgs::msg::OccupancyGrid grid_msg;

  // Header
  grid_msg.header.stamp = this->now();
  grid_msg.header.frame_id = world_frame_id_;

  // Map meta-data
  grid_msg.info.resolution = bloomxai_->getResolution();
  grid_msg.info.width = map_size[0];
  grid_msg.info.height = map_size[1];
  grid_msg.info.origin.position.x = min[0];
  grid_msg.info.origin.position.y = min[1];
  grid_msg.info.origin.position.z = 0.0;
  grid_msg.info.origin.orientation.w = 1.0;

  // Flatten the matrix into a 1D occupancy data array
  grid_msg.data.resize(map_size[0] * map_size[1]);
  if (matrix.size() != static_cast<size_t>(map_size[1]) ||
      (matrix.size() > 0 && matrix[0].size() != static_cast<size_t>(map_size[0]))) {
    std::cerr << "Error: map_size does not match matrix dimensions\n";
    return false;
  }
  for (int y = 0; y < map_size[1]; y++) {
    for (int x = 0; x < map_size[0]; x++) {
      // std::cout << " Coords : " << x << "," << y << " Size: " << map_size[0] << "," <<
      // map_size[1] << std::endl;

      int idx = y * map_size[0] + x;
      int val = matrix[y][x];
      grid_msg.data[idx] = val;
    }
  }
  std::cout << "Publishing grid map" << std::endl;
  grid_map_pub_->publish(grid_msg);
  return true;
}

bool BloomxaiServer::savePGMMapCallback(
    const std::shared_ptr<bloomxai_ros::srv::SaveMap::Request> request,
    const std::shared_ptr<bloomxai_ros::srv::SaveMap::Response> response) {
  try {
    RCLCPP_INFO(get_logger(), "Saving PGM map to %s", request->filename.c_str());

    auto filename = request->filename;
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open()) {
      std::cout << "Could not open file " << filename << std::endl;
      return false;
    }
    std::cout << "Saving PGM map" << std::endl;

    float th_z = 0.5;

    // We have to project the map on the x-y plane. We will create a matrix for that with
    // the map bbox in x, y and z.
    std::vector<float> min, max;
    bloomxai_->getMapLimits(min, max);
    std::vector<int> map_size = bloomxai_->getMapXYSize();
    // To round correctly, we have to rest half of the resolution to the max
    for (int i = 0; i < 3; i++) {
      max[i] -= bloomxai_->getResolution() / 2.0;
    }

    file << "P2\n"
         << "# Created by VoxelHashMap\n"
         << map_size[0] << " " << map_size[1] << "\n"
         << 255 << std::endl;

    std::cout << "Detected limits: X [" << min[0] << "," << max[0] << "] Y [" << min[1] << ","
              << max[1] << "] Z [" << min[2] << "," << max[2] << "]" << std::endl;
    std::cout << "Resolution: " << bloomxai_->getResolution() << std::endl;
    std::cout << "Map size: [" << map_size[0] << "," << map_size[1] << "]" << std::endl;

    auto matrix = bloomxai_->generate2DGridMap(min, max, map_size, problematic_classes_, task_classes_, th_z);

    for (int i = map_size[1] - 1; i >= 0; i--) {
      for (int j = 0; j < map_size[0]; j++) {
        int value = matrix[i][j];
        if (value == 0) {
          value = 255;
        } else if (value == 51) {
          value = 80;
        } else if (value == 22) {
          value = 200;
        } else {
          value = 0;
        }
        file << value << " ";
      }
      file << std::endl;
    }

    file.close();

    RCLCPP_INFO(get_logger(), "Map saved to %s", request->filename.c_str());
    response->success = true;
    response->message = "Map saved successfully.";
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Failed to save map: %s", e.what());
    response->success = false;
    response->message = e.what();
    return false;
  }
}

bool BloomxaiServer::loadMapCallback(
    const std::shared_ptr<bloomxai_ros::srv::LoadMap::Request> request,
    const std::shared_ptr<bloomxai_ros::srv::LoadMap::Response> response) {
  try {
    bloomxai_->deserializeFromFile(request->filename);
    RCLCPP_INFO(get_logger(), "Map loaded from %s", request->filename.c_str());
    response->success = true;
    response->message = "Map loaded successfully.";
    // publishAll(now());
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Failed to load map: %s", e.what());
    response->success = false;
    response->message = e.what();
    return false;
  }
}

void BloomxaiServer::insertCloudCallback(const PointCloud2::ConstSharedPtr cloud) {
  const auto start = rclcpp::Clock{}.now();

  RCLCPP_INFO(get_logger(), "Received cloud with %d points", cloud->width * cloud->height);
  // Sensor In Global Frames Coordinates
  geometry_msgs::msg::TransformStamped sensor_to_world_transform_stamped;
  try {
    sensor_to_world_transform_stamped = tf2_buffer_->lookupTransform(
        world_frame_id_, cloud->header.frame_id, cloud->header.stamp,
        rclcpp::Duration::from_seconds(5.0));
  } catch (const tf2::TransformException& ex) {
    RCLCPP_WARN(this->get_logger(), "%s", ex.what());
    return;
  }

  PCLPointCloud pc;  // input cloud for filtering and ground-detection
  std::vector<Bloomxai::VSemantics> semantics;

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud, "z");
  sensor_msgs::PointCloud2ConstIterator<uint32_t> iter_rgba(*cloud, "rgb");
  sensor_msgs::PointCloud2ConstIterator<float> iter_gt_semantics(*cloud, "gt_semantics");
  sensor_msgs::PointCloud2ConstIterator<float> iter_semantics(*cloud, "semantics");

  float prob_max = 0.0;
  float prob_min = 1.0;

  // Temporary to track original index for filtering semantics later
  std::vector<int> original_indices;
  int point_index = 0;

  for (; iter_x != iter_x.end();
       ++iter_x, ++iter_y, ++iter_z, ++iter_rgba, ++iter_gt_semantics, ++iter_semantics) {
    pcl::PointXYZ p;
    p.x = *iter_x;
    p.y = *iter_y;
    p.z = *iter_z;

    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        (p.x == 0.0 && p.y == 0.0 && p.z == 0.0) || p.x < min_range_) {
      continue;
    }

    pc.points.push_back(p);
    original_indices.push_back(point_index);

    Bloomxai::VSemantics semantic(sem_dim_);
    for (int i = 0; i < sem_dim_; ++i) {
      semantic[i] = iter_semantics[i];
      if (semantic[i] > prob_max) {
        prob_max = semantic[i];
      }
      if (semantic[i] < prob_min) {
        prob_min = semantic[i];
      }
    };
    semantics.push_back(semantic);

    point_index++;
  }

  // === Apply Radius Outlier Removal ===
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
  outrem.setInputCloud(pc.makeShared());
  outrem.setRadiusSearch(0.1);        // Radius in meters
  outrem.setMinNeighborsInRadius(5);  // Keep points with >= 5 neighbors

  PCLPointCloud pc_filtered;
  std::vector<int> kept_indices;

  outrem.filter(kept_indices);
  pcl::copyPointCloud(pc, kept_indices, pc_filtered);

  // Filter the semantics array to match filtered point cloud
  std::vector<Bloomxai::VSemantics> filtered_semantics;
  for (const auto& idx : kept_indices) {
    filtered_semantics.push_back(semantics[idx]);
  }

  pc = std::move(pc_filtered);
  semantics = std::move(filtered_semantics);

  RCLCPP_INFO(get_logger(), "Filtered cloud has %lu points", pc.points.size());
  RCLCPP_INFO(
      get_logger(), "Semantics sanity check: Max prob: %f, Min prob: %f", prob_max, prob_min);

  Eigen::Matrix4f sensor_to_world =
      tf2::transformToEigen(sensor_to_world_transform_stamped.transform).matrix().cast<float>();

  // Transforming Points to Global Reference Frame
  pcl::transformPointCloud(pc, pc, sensor_to_world);

  // Getting the Translation from the sensor to the Global Reference Frame
  const auto& t = sensor_to_world_transform_stamped.transform.translation;

  RCLCPP_INFO(get_logger(), "Inserting %ld points", pc.points.size());

  const pcl::PointXYZ sensor_to_world_vec3(t.x, t.y, t.z);

  {
    std::lock_guard<std::mutex> lock(bloomxai_mutex_);

    if (max_range_ >= 0) {
      bloomxai_->insertPointCloud(pc.points, semantics, sensor_to_world_vec3, max_range_);
    } else {
      bloomxai_->insertPointCloud(
          pc.points, semantics, sensor_to_world_vec3, std::numeric_limits<double>::infinity());
    }
  }

  double total_elapsed = (rclcpp::Clock{}.now() - start).seconds();
  RCLCPP_INFO(get_logger(), "Pointcloud insertion in Bonxai done, %f sec)", total_elapsed);

  // publishAll();
}

rcl_interfaces::msg::SetParametersResult BloomxaiServer::onParameter(
    const std::vector<rclcpp::Parameter>& parameters) {
  update_param(parameters, "occupancy_min_z", occupancy_min_z_);
  update_param(parameters, "occupancy_max_z", occupancy_max_z_);

  double sensor_model_min{get_parameter("sensor_model.min").as_double()};
  update_param(parameters, "sensor_model.min", sensor_model_min);
  double sensor_model_max{get_parameter("sensor_model.max").as_double()};
  update_param(parameters, "sensor_model.max", sensor_model_max);
  double sensor_model_hit{get_parameter("sensor_model.hit").as_double()};
  update_param(parameters, "sensor_model.hit", sensor_model_hit);
  double sensor_model_miss{get_parameter("sensor_model.miss").as_double()};
  update_param(parameters, "sensor_model.miss", sensor_model_miss);

  BloomxaiT::Options options;
  options.prob_miss_log = bloomxai_->logods(sensor_model_miss);
  options.prob_hit_log = bloomxai_->logods(sensor_model_hit);
  options.clamp_min_log = bloomxai_->logods(sensor_model_min);
  options.clamp_max_log = bloomxai_->logods(sensor_model_max);

  // Check semantics

  if (semantic_type_ == SemanticType::PROBABILITIES) {
    // Probabilities
    double alpha_reg{get_parameter("semantics.alpha_reg").as_double()};
    update_param(parameters, "semantics.alpha_reg", alpha_reg);
    double clamp_min_prob{get_parameter("semantics.clamp_min_prob").as_double()};
    update_param(parameters, "semantics.clamp_min_prob", clamp_min_prob);
    double clamp_max_prob{get_parameter("semantics.clamp_max_prob").as_double()};
    update_param(parameters, "semantics.clamp_max_prob", clamp_max_prob);
    Bloomxai::ProbabilitySemanticOperator::ProbOptions prob_options;
    prob_options.alpha_reg = alpha_reg;
    prob_options.clamp_min_prob = clamp_min_prob;
    prob_options.clamp_max_prob = clamp_max_prob;
    bloomxai_->semantic_operator->setOptions(prob_options);
  } else if (semantic_type_ == SemanticType::FEATURES) {
    // Features
    // double occ_thres = get_parameter("semantics.occ_thres").as_double();
    // update_param(parameters, "semantics.occ_thres", occ_thres);
    int sim_query_index = get_parameter("semantics.sim_query_index").as_int();
    update_param(parameters, "semantics.sim_query_index", sim_query_index);
    sim_query_index_ = sim_query_index;
    // Bloomxai::FeatureSemanticOperator::FeatOptions feat_options;
    // feat_options.occ_thres = occ_thres;
    // bloomxai_->semantic_operator->setOptions(feat_options);
  }

  std::lock_guard<std::mutex> lock(bloomxai_mutex_);
  bloomxai_->setOptions(options);

  // publishAll(now());

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  return result;
}

void BloomxaiServer::publishAll() {

  auto start = std::chrono::steady_clock::now();
  const auto rostime = this->now();
  thread_local std::vector<Eigen::Vector3d> bloomxai_result;
  bloomxai_result.clear();
  thread_local std::vector<int> labels;
  thread_local std::vector<float> similarities;
  labels.clear();
  similarities.clear();
  {
    std::lock_guard<std::mutex> lock(bloomxai_mutex_);
    if (!bloomxai_->semantic_operator->isReady()) {
      std::cout << "Semantics are not ready for publishing" << std::endl;
      return;
    }
    bloomxai_->getOccupiedVoxelsClassAndSimilarity(
        sim_query_index_, bloomxai_result, labels, similarities);
  }

  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
  RCLCPP_INFO(get_logger(), "Getting data took %d ms", elapsed);

  if (bloomxai_result.size() <= 1) {
    RCLCPP_WARN(get_logger(), "Nothing to publish, bloomxai is empty");
    return;
  }

  // Publish Cube Marker
  thread_local visualization_msgs::msg::Marker marker;
  marker.header.frame_id = world_frame_id_;
  marker.header.stamp = rostime;
  marker.ns = "bloomxai_voxels";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = res_;  // size of voxel
  marker.scale.y = res_;
  marker.scale.z = res_;
  marker.points.clear();
  marker.colors.clear();

  for (size_t i = 0; i < bloomxai_result.size(); ++i) {
    const auto& voxel = bloomxai_result[i];
    if (voxel.z() >= occupancy_min_z_ && voxel.z() <= occupancy_max_z_) {
      geometry_msgs::msg::Point p;
      p.x = voxel.x();
      p.y = voxel.y();
      p.z = voxel.z();
      marker.points.push_back(p);

      std_msgs::msg::ColorRGBA color;
      auto rgb = label_to_rgb_[labels[i]];
      color.r = rgb[0] / 255.0f;
      color.g = rgb[1] / 255.0f;
      color.b = rgb[2] / 255.0f;
      color.a = 1.0f;
      marker.colors.push_back(color);
    }
  }

  if (marker_pub_->get_subscription_count() + marker_pub_->get_intra_process_subscription_count() >
      0) {
    marker_pub_->publish(marker);
    RCLCPP_INFO(get_logger(), "Published marker with %ld cubes", marker.points.size());
  }

  if (grid_map_pub_->get_subscription_count() +
          grid_map_pub_->get_intra_process_subscription_count() >
      0) {
    sendGridMap();
  }

  if (semantic_type_ == SemanticType::FEATURES) {
    // Publish similarity
    // Publish Cube Marker
    thread_local visualization_msgs::msg::Marker marker;
    marker.header.frame_id = world_frame_id_;
    marker.header.stamp = rostime;
    marker.ns = "similarity_voxels";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = res_;  // size of voxel
    marker.scale.y = res_;
    marker.scale.z = res_;
    marker.points.clear();
    marker.colors.clear();

    // --- Phase 1: Gather Valid Voxels and Find Max Similarity ---
    std::vector<geometry_msgs::msg::Point> valid_points;
    std::vector<float> valid_similarities;
    float max_similarity = 0.0f;

    for (size_t i = 0; i < bloomxai_result.size(); ++i) {
      const auto& voxel = bloomxai_result[i];
      float current_similarity = similarities[i];

      if (voxel.z() >= occupancy_min_z_ && voxel.z() <= occupancy_max_z_) {
        // Store the point coordinates
        geometry_msgs::msg::Point p;
        p.x = voxel.x();
        p.y = voxel.y();
        p.z = voxel.z();
        valid_points.push_back(p);

        // Store the similarity value
        valid_similarities.push_back(current_similarity);

        // Update the maximum similarity found so far
        if (current_similarity > max_similarity) {
          max_similarity = current_similarity;
        }
      }
    }

    if (max_similarity == 0.0f) {
      // Exit the scope if no voxels are valid or max similarity is zero.
      // The marker remains empty.
      return;
    }

    // --- Phase 2: Normalize and Visualize ---

    for (size_t i = 0; i < valid_points.size(); ++i) {
      // 1. Normalize the similarity value
      float normalized_sim = valid_similarities[i] / max_similarity;

      // 2. Add the point to the marker
      marker.points.push_back(valid_points[i]);

      // 3. Calculate and add color
      std_msgs::msg::ColorRGBA color;
      // std::cout << normalized_sim << std::endl; // Use normalized value here
      auto rgb = sim_to_rgb_(normalized_sim);

      color.r = rgb[0] / 255.0f;
      color.g = rgb[1] / 255.0f;
      color.b = rgb[2] / 255.0f;
      color.a = 1.0f;
      marker.colors.push_back(color);
    }

    if (marker_pub_->get_subscription_count() +
            marker_pub_->get_intra_process_subscription_count() >
        0) {
      marker_pub_->publish(marker);
      RCLCPP_INFO(get_logger(), "Published marker with %ld cubes", marker.points.size());
    }
  }
}

bool BloomxaiServer::resetSrv(
    const std::shared_ptr<ResetSrv::Request>, const std::shared_ptr<ResetSrv::Response>) {
  const auto rostime = now();
  std::lock_guard<std::mutex> lock(bloomxai_mutex_);
  // Reinitialize the operator
  std::unique_ptr<Bloomxai::BaseSemanticOperator> semantic_operator;
  if (semantic_type_ == SemanticType::PROBABILITIES) {
    semantic_operator = std::make_unique<Bloomxai::ProbabilitySemanticOperator>(sem_dim_);
  } else if (semantic_type_ == SemanticType::FEATURES) {
    semantic_operator = std::make_unique<Bloomxai::FeatureSemanticOperator>(sem_dim_);
  }
  bloomxai_ = std::make_unique<BloomxaiT>(res_, sem_dim_, std::move(semantic_operator));

  RCLCPP_INFO(get_logger(), "Cleared Bonxai");
  // publishAll(rostime);

  return true;
}

}  // namespace bloomxai_server

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(bloomxai_server::BloomxaiServer)