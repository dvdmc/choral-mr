# This file is inspired by: https://github.com/ros-navigation/navigation2/blob/main/nav2_bringup/launch/bringup_launch.py
import os

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
)
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import (
    Node,
)
from launch.substitutions import (
    PathJoinSubstitution,
    LaunchConfiguration,
    Command,
)
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():

    # Paths to bringup and custom package
    tb2_system_dir = FindPackageShare("tb2_system")

    # Declare launch arguments
    declarenamespace_cmd = DeclareLaunchArgument(
        "namespace", default_value="ugv0", description="Namespace for the robot"
    )
    declare_use_namespace_cmd = DeclareLaunchArgument(
        "use_namespace",
        default_value="False",
        description="Whether to apply a namespace to the navigation stack",
    )
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="False",
        description="Use simulation (Gazebo) clock if True",
    )
    declare_urdf_file_cmd = DeclareLaunchArgument(
        "urdf_file",
        default_value=PathJoinSubstitution([tb2_system_dir, "urdf", "turtlebot_hokuyo_top.urdf.xacro"]),
        description="URDF file for the robot",
    )

    # Locate description
    # TODO: Fix the xacro compilation  https://gist.github.com/clalancette/5d15df1f54a1e01946659dbfa6c46c30
    # We currently use the compiled urdf using: xacro -o compiled_turtlebot.urdf turtlebot.urdf.xacro
    robot_description = Command(["xacro ", LaunchConfiguration("urdf_file")])

    start_robot_state_publisher_cmd = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        # namespace=namespace,
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time"), "robot_description": ParameterValue(robot_description, value_type=str)}
        ],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
    )

    return LaunchDescription(
        [
            declarenamespace_cmd,
            declare_use_namespace_cmd,
            declare_use_sim_time_cmd,
            declare_urdf_file_cmd,
            start_robot_state_publisher_cmd,
        ]
    )
