#include "mr_het_coord_ros/mr_het_planner_node.hpp"

MRPlannerNode::MRPlannerNode() : Node("mr_het_planner_node") {
  readROSParameters();

  for (size_t i = 0; i < agent_types_.size(); i++) {
    if (agent_types_[i] == AgentType::GROUND) {
      // TODO: Namespacing in Nav2 will be handled in next ROS2 release Add: "/"
      // + robot_ns_[i] +
      ugv_client_ptrs_.push_back(rclcpp_action::create_client<UGVFollowPath>(
          this, "/follow_waypoints"));
      global_to_local_id_map_.push_back(ugv_client_ptrs_.size() - 1);
    } else if (agent_types_[i] == AgentType::AERIAL) {
      uav_client_ptrs_.push_back(rclcpp_action::create_client<DroneFollowPath>(
          this, "/" + robot_ns_[i] + "/FollowPathBehavior"));
      global_to_local_id_map_.push_back(uav_client_ptrs_.size() - 1);
    }
  }

  sub_plans_ = this->create_subscription<mr_het_coord_ros::msg::VrpSolution>(
      "/vrp_sol", 10,
      std::bind(&MRPlannerNode::pathsCallback, this, std::placeholders::_1));
}

void MRPlannerNode::readROSParameters() {
  this->declare_parameter("robot_ns", std::vector<std::string>({}));
  this->declare_parameter("agent_types", std::vector<std::string>({}));
  this->declare_parameter("global_frame", "map");

  this->get_parameter("robot_ns", robot_ns_);
  std::vector<std::string> agent_types_str;
  this->get_parameter("agent_types", agent_types_str);
  this->get_parameter("global_frame", global_frame_);

  for (auto s : agent_types_str) {
    agent_types_.push_back(stringToAgentType(s));
  }
  if (agent_types_.size() != robot_ns_.size()) {
    RCLCPP_ERROR(
        this->get_logger(),
        "Number of robot namespaces does not match number of agent types");
    exit(1);
  }

  // Log parameters
  RCLCPP_INFO(this->get_logger(), "Robot namespaces:");
  for (auto ns : robot_ns_) {
    RCLCPP_INFO(this->get_logger(), "- %s", ns.c_str());
  }
  RCLCPP_INFO(this->get_logger(), "Agent types:");
  for (auto t : agent_types_) {
    RCLCPP_INFO(this->get_logger(), "- %s", agentTypeToString(t).c_str());
  }
  RCLCPP_INFO(this->get_logger(), "Global frame: %s", global_frame_.c_str());
}

void MRPlannerNode::pathsCallback(
    const mr_het_coord_ros::msg::VrpSolution::SharedPtr msg) {
  RCLCPP_INFO(this->get_logger(), "Received new VRP solution");
  size_t num_agents = msg->num_agents;
  if (num_agents != agent_types_.size()) {
    RCLCPP_ERROR(this->get_logger(),
                 "Number of paths does not match number of agents");
    return;
  }
  for (size_t i = 0; i < num_agents; i++) {
    if (agent_types_[i] == AgentType::GROUND) {
      sendUGVAction(msg->paths[i], i);
    } else if (agent_types_[i] == AgentType::AERIAL) {
      sendUAVAction(msg->paths[i], i);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Unknown agent type");
      return;
    }
  }
}

geometry_msgs::msg::PoseArray MRPlannerNode::correctGoals(
    const geometry_msgs::msg::PoseArray& path) {
  geometry_msgs::msg::PoseArray corrected_path = path;
  // Since we don't want to crash into obstacles, we stop a bit before the goal
  float safety_distance = 0.3;  // meters
  if (corrected_path.poses.size() < 1) {
    // Don't have to correct anything
    return corrected_path;
  }

  geometry_msgs::msg::Pose current_pose = corrected_path.poses[0];
  for (size_t i = 1; i < corrected_path.poses.size(); i++) {
    geometry_msgs::msg::Pose next_pose = corrected_path.poses[i];

    double dx = next_pose.position.x - current_pose.position.x;
    double dy = next_pose.position.y - current_pose.position.y;
    double dz = next_pose.position.z - current_pose.position.z;
    double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < safety_distance) {
      // Too close, no correction needed
      continue;
    }
    double correction_ratio = (dist - safety_distance) / dist;

    geometry_msgs::msg::Pose corrected_pose;
    corrected_pose.position.x = current_pose.position.x + dx * correction_ratio;
    corrected_pose.position.y = current_pose.position.y + dy * correction_ratio;
    corrected_pose.position.z = current_pose.position.z + dz * correction_ratio;
    corrected_pose.orientation = next_pose.orientation;

    corrected_path.poses[i] = corrected_pose;
    current_pose = corrected_pose;
  }

  return corrected_path;
}

geometry_msgs::msg::Quaternion MRPlannerNode::get_orientation_from_points(
    const geometry_msgs::msg::Point& current,
    const geometry_msgs::msg::Point& next) {
  // Calculate the angle (yaw) using atan2(delta_y, delta_x)
  double delta_x = next.x - current.x;
  double delta_y = next.y - current.y;
  double yaw = std::atan2(delta_y, delta_x);

  // Convert yaw (single angle) to a Quaternion (x, y, z, w)
  tf2::Quaternion q;
  q.setRPY(0, 0, yaw);  // Roll, Pitch, Yaw

  return tf2::toMsg(q);
}

void MRPlannerNode::sendUGVAction(geometry_msgs::msg::PoseArray const& path,
                                  int id) {
  int local_id = global_to_local_id_map_[id];
  if (!ugv_client_ptrs_[local_id]->wait_for_action_server(
          std::chrono::milliseconds(1000))) {
    RCLCPP_ERROR(this->get_logger(), "UGV action server not available");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Sending UGV goal to %s",
              robot_ns_[id].c_str());

  auto corrected_path = correctGoals(path);

  auto goal_msg = UGVFollowPath::Goal();
  std::vector<geometry_msgs::msg::PoseStamped> poses_msg;

  for (size_t i = 0; i < corrected_path.poses.size(); i++) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = global_frame_;
    pose.header.stamp = this->now();
    pose.pose.position =
        corrected_path.poses[i].position;  // Only copy position (X, Y, Z)

    if (i < corrected_path.poses.size() - 1) {
      // For intermediate poses, point the robot toward the next pose
      pose.pose.orientation =
          get_orientation_from_points(corrected_path.poses[i].position,
                                      corrected_path.poses[i + 1].position);
    } else {
      pose.pose.orientation = corrected_path.poses[i].orientation;
    }

    poses_msg.push_back(pose);
  }

  goal_msg.poses = poses_msg;

  auto send_goal_options =
      rclcpp_action::Client<UGVFollowPath>::SendGoalOptions();
  send_goal_options.goal_response_callback = std::bind(
      &MRPlannerNode::ugv_goal_response_callback, this, std::placeholders::_1);
  send_goal_options.feedback_callback =
      std::bind(&MRPlannerNode::ugv_feedback_callback, this,
                std::placeholders::_1, std::placeholders::_2);
  send_goal_options.result_callback = std::bind(
      &MRPlannerNode::ugv_result_callback, this, std::placeholders::_1);

  this->ugv_client_ptrs_[local_id]->async_send_goal(goal_msg,
                                                    send_goal_options);
}

void MRPlannerNode::sendUAVAction(geometry_msgs::msg::PoseArray const& path,
                                  int id) {
  int local_id = global_to_local_id_map_[id];
  if (!uav_client_ptrs_[local_id]->wait_for_action_server(
          std::chrono::milliseconds(1000))) {
    RCLCPP_ERROR(this->get_logger(), "Drone action server not available");
    return;
  }
  RCLCPP_INFO(this->get_logger(), "Sending UAV goal to %s",
              robot_ns_[id].c_str());

  auto corrected_path = correctGoals(path);

  auto uav_goal_msg = DroneFollowPath::Goal();
  uav_goal_msg.header.frame_id = global_frame_;
  uav_goal_msg.max_speed = 0.5;

  for (size_t i = 0; i < corrected_path.poses.size(); i++) {
    as2_msgs::msg::PoseWithID pose;
    pose.id = i;
    pose.pose = corrected_path.poses[i];
    uav_goal_msg.path.push_back(pose);
  }

  auto uav_send_goal_options =
      rclcpp_action::Client<DroneFollowPath>::SendGoalOptions();
  uav_send_goal_options.goal_response_callback = std::bind(
      &MRPlannerNode::uav_goal_response_callback, this, std::placeholders::_1);
  uav_send_goal_options.feedback_callback =
      std::bind(&MRPlannerNode::uav_feedback_callback, this,
                std::placeholders::_1, std::placeholders::_2);
  uav_send_goal_options.result_callback = std::bind(
      &MRPlannerNode::uav_result_callback, this, std::placeholders::_1);

  this->uav_client_ptrs_[local_id]->async_send_goal(uav_goal_msg,
                                                    uav_send_goal_options);
}

// UGV callbacks
void MRPlannerNode::ugv_goal_response_callback(
    const GoalHandleUGVFollowPath::SharedPtr& goal_handle) {
  if (!goal_handle) {
    RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
  } else {
    RCLCPP_INFO(this->get_logger(),
                "Goal accepted by server, waiting for result");
  }
}
void MRPlannerNode::ugv_feedback_callback(
    GoalHandleUGVFollowPath::SharedPtr,
    const std::shared_ptr<const UGVFollowPath::Feedback> feedback) {
  std::stringstream ss;
  ss << "Current waypoint: " << feedback->current_waypoint << "\n";
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());
}
void MRPlannerNode::ugv_result_callback(
    const GoalHandleUGVFollowPath::WrappedResult& result) {
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      break;
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
      return;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
      return;
    default:
      RCLCPP_ERROR(this->get_logger(), "Unknown result code");
      return;
  }
  std::stringstream ss;
  ss << "Goal reached! ";
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());
}

// UAV callbacks
void MRPlannerNode::uav_goal_response_callback(
    const GoalHandleDroneFollowPath::SharedPtr& goal_handle) {
  if (!goal_handle) {
    RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
  } else {
    RCLCPP_INFO(this->get_logger(),
                "Goal accepted by server, waiting for result");
  }
}
void MRPlannerNode::uav_feedback_callback(
    GoalHandleDroneFollowPath::SharedPtr,
    const std::shared_ptr<const DroneFollowPath::Feedback> feedback) {
  std::stringstream ss;
  ss << "Distance to the goal: " << feedback->actual_distance_to_next_waypoint
     << "\n";
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());
}
void MRPlannerNode::uav_result_callback(
    const GoalHandleDroneFollowPath::WrappedResult& result) {
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      break;
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
      return;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
      return;
    default:
      RCLCPP_ERROR(this->get_logger(), "Unknown result code");
      return;
  }
  std::stringstream ss;
  ss << "Goal reached! ";
  RCLCPP_INFO(this->get_logger(), ss.str().c_str());
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MRPlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}