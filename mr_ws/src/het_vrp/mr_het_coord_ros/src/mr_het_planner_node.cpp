#include "mr_het_coord_ros/mr_het_planner_node.hpp"

MRPlannerNode::MRPlannerNode() : Node("mr_het_planner_node")
{
    readROSParameters();

    for (int i = 0; i < agent_types_.size(); i++)
    {
        if (agent_types_[i] == AgentType::GROUND)
        {
            std::string action_name = "/" + robot_ns_[i] + "/navigate_through_poses";
            ugv_client_ptrs_.push_back(
                rclcpp_action::create_client<UGVFollowPath>(this, action_name));
        }
        else if (agent_types_[i] == AgentType::AERIAL)
        {
            std::string action_name = "/" + robot_ns_[i] + "/FollowPathBehavior";
            uav_client_ptrs_.push_back(
                rclcpp_action::create_client<DroneFollowPath>(this, action_name));
        }
    }
}

void MRPlannerNode::readROSParameters()
{
    this->declare_parameter("robot_ns", std::vector<std::string>({}));
    this->declare_parameter("agent_types", std::vector<std::string>({}));

    this->get_parameter("robot_ns", robot_ns_);
    std::vector<std::string> agent_types_str;
    this->get_parameter("agent_types", agent_types_str);

    for (auto s : agent_types_str)
    {
        agent_types_.push_back(stringToAgentType(s));
    }
    if (agent_types_.size() != robot_ns_.size())
    {
        RCLCPP_ERROR(this->get_logger(), "Number of robot namespaces does not match number of agent types");
        exit(1);
    }
}

void MRPlannerNode::pathsCallback(const mr_het_coord_ros::msg::VrpSolution::SharedPtr msg)
{
    int num_agents = msg->num_agents;
    if (num_agents != agent_types_.size())
    {
        RCLCPP_ERROR(this->get_logger(), "Number of paths does not match number of agents");
        return;
    }
    for (int i = 0; i < num_agents; i++)
    {
        if (agent_types_[i] == AgentType::GROUND)
        {
            sendUGVAction(msg->paths[i], i);
        }
        else if(agent_types_[i] == AgentType::AERIAL)
        {
            sendUAVAction(msg->paths[i], i);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unknown agent type");
            return;
        }
    }
}

void MRPlannerNode::sendUGVAction(geometry_msgs::msg::PoseArray const &path, int i)
{
    if (!ugv_client_ptrs_[i]->wait_for_action_server())
    {
        RCLCPP_ERROR(this->get_logger(), "UGV action server not available");
        rclcpp::shutdown();
    }

    auto goal_msg = UGVFollowPath::Goal();
    std::vector<geometry_msgs::msg::PoseStamped> poses_msg;

    for (int i = 0; i < path.poses.size(); i++)
    {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.header.stamp = this->now();
        pose.pose = path.poses[i];
        poses_msg.push_back(pose);
    }

    goal_msg.poses = poses_msg;
    //   RCLCPP_INFO(this->get_logger(), "Sending goal");
    //   for(int i = 0; i < path.size(); i++) {
    //     std::stringstream ss;
    //     ss << path[i][0] << " " << path[i][1] << " " << path[i][2];
    //     RCLCPP_INFO(this->get_logger(), ss.str().c_str());
    //   }
    auto send_goal_options = rclcpp_action::Client<UGVFollowPath>::SendGoalOptions();
    send_goal_options.goal_response_callback =
        std::bind(&MRPlannerNode::ugv_goal_response_callback, this, std::placeholders::_1);
    send_goal_options.feedback_callback =
        std::bind(&MRPlannerNode::ugv_feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
    send_goal_options.result_callback =
        std::bind(&MRPlannerNode::ugv_result_callback, this, std::placeholders::_1);

    this->ugv_client_ptrs_[i]->async_send_goal(goal_msg, send_goal_options);
}

void MRPlannerNode::sendUAVAction(geometry_msgs::msg::PoseArray const &path, int i)
{
    if (!uav_client_ptrs_[i]->wait_for_action_server())
    {
        RCLCPP_ERROR(this->get_logger(), "Drone action server not available");
        rclcpp::shutdown();
    }

    auto uav_goal_msg = DroneFollowPath::Goal();
    uav_goal_msg.header.frame_id = "map";
    uav_goal_msg.max_speed = 0.5;

    for (int i = 0; i < path.poses.size(); i++)
    {
        as2_msgs::msg::PoseWithID pose;
        pose.id = i;
        pose.pose = path.poses[i];
        uav_goal_msg.path.push_back(pose);
    }

    auto uav_send_goal_options = rclcpp_action::Client<DroneFollowPath>::SendGoalOptions();
    uav_send_goal_options.goal_response_callback =
        std::bind(&MRPlannerNode::uav_goal_response_callback, this, std::placeholders::_1);
    uav_send_goal_options.feedback_callback =
        std::bind(&MRPlannerNode::uav_feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
    uav_send_goal_options.result_callback =
        std::bind(&MRPlannerNode::uav_result_callback, this, std::placeholders::_1);

    this->uav_client_ptrs_[i]->async_send_goal(uav_goal_msg, uav_send_goal_options);
}

// UGV callbacks
void MRPlannerNode::ugv_goal_response_callback(const GoalHandleUGVFollowPath::SharedPtr &goal_handle)
{
    if (!goal_handle)
    {
        RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
    }
}
void MRPlannerNode::ugv_feedback_callback(
    GoalHandleUGVFollowPath::SharedPtr,
    const std::shared_ptr<const UGVFollowPath::Feedback> feedback)
{
    std::stringstream ss;
    ss << "Distance to the goal: " << feedback->distance_remaining << "\n";
    RCLCPP_INFO(this->get_logger(), ss.str().c_str());
}
void MRPlannerNode::ugv_result_callback(const GoalHandleUGVFollowPath::WrappedResult &result)
{
    switch (result.code)
    {
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
void MRPlannerNode::uav_goal_response_callback(const GoalHandleDroneFollowPath::SharedPtr &goal_handle)
{
    if (!goal_handle)
    {
        RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
    }
}
void MRPlannerNode::uav_feedback_callback(
    GoalHandleDroneFollowPath::SharedPtr,
    const std::shared_ptr<const DroneFollowPath::Feedback> feedback)
{
    std::stringstream ss;
    ss << "Distance to the goal: " << feedback->actual_distance_to_next_waypoint << "\n";
    RCLCPP_INFO(this->get_logger(), ss.str().c_str());
}
void MRPlannerNode::uav_result_callback(const GoalHandleDroneFollowPath::WrappedResult &result)
{
    switch (result.code)
    {
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

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MRPlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}