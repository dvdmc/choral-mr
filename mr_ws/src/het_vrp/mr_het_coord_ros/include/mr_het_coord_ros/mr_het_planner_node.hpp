#ifndef MR_HET_COORD_ROS_PLANNER_NODE_HPP
#define MR_HET_COORD_ROS_PLANNER_NODE_HPP

#include <functional>
#include <future>
#include <memory>
#include <sstream>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "nav_msgs/msg/path.hpp"
#include "tf2/LinearMath/Quaternion.h"

#include "as2_msgs/action/follow_path.hpp"
#include "as2_msgs/msg/pose_with_id.hpp"
#include "nav2_msgs/action/navigate_through_poses.hpp"

#include "mr_het_coord/map/map.hpp" // For AgentType
#include "mr_het_coord_ros/msg/vrp_solution.hpp"

class MRPlannerNode : public rclcpp::Node
{
public:
    MRPlannerNode();
    ~MRPlannerNode() {};

    using DroneFollowPath = as2_msgs::action::FollowPath;
    using GoalHandleDroneFollowPath = rclcpp_action::ClientGoalHandle<DroneFollowPath>;
    using UGVFollowPath = nav2_msgs::action::NavigateThroughPoses;
    using GoalHandleUGVFollowPath = rclcpp_action::ClientGoalHandle<UGVFollowPath>;
    
private:
    rclcpp::Subscription<mr_het_coord_ros::msg::VrpSolution>::SharedPtr sub_map_;
    std::vector<rclcpp_action::Client<UGVFollowPath>::SharedPtr> ugv_client_ptrs_;
    std::vector<rclcpp_action::Client<DroneFollowPath>::SharedPtr> uav_client_ptrs_;

    std::vector<std::string> robot_ns_;
    std::vector<AgentType> agent_types_;

    void readROSParameters();
    
    void pathsCallback(const mr_het_coord_ros::msg::VrpSolution::SharedPtr msg);
    void sendUGVAction(geometry_msgs::msg::PoseArray const &path, int i);
    void sendUAVAction(geometry_msgs::msg::PoseArray const &path, int i);

    // UGV callbacks
    void ugv_goal_response_callback(const GoalHandleUGVFollowPath::SharedPtr & goal_handle);
    void ugv_feedback_callback(
        GoalHandleUGVFollowPath::SharedPtr,
        const std::shared_ptr<const UGVFollowPath::Feedback> feedback);
    void ugv_result_callback(const GoalHandleUGVFollowPath::WrappedResult & result);

    // UAV callbacks
    void uav_goal_response_callback(const GoalHandleDroneFollowPath::SharedPtr & goal_handle);
    void uav_feedback_callback(
        GoalHandleDroneFollowPath::SharedPtr,
        const std::shared_ptr<const DroneFollowPath::Feedback> feedback);
    void uav_result_callback(const GoalHandleDroneFollowPath::WrappedResult & result);
};

#endif