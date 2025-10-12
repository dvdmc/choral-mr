from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import (LaunchConfiguration, PythonExpression, PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.substitutions import (
    FindPackageShare,
)


def generate_launch_description():
    # Declare the 'map' launch argument.
    map_arg = DeclareLaunchArgument(
        "map", description="Map name for the configuration file"
    )
    num_agents_arg = DeclareLaunchArgument(
        "num_agents", description="Number of agents for the configuration file"
    )

    # Load the config file
    mr_coord_node_parameters = [
        PathJoinSubstitution(
            [
                FindPackageShare("mr_het_coord_ros"),
                "cfg",
                PythonExpression(
                    ["'config_' + '", LaunchConfiguration("map"), "' + '.yml'"]
                ),
            ]
        ),
        PathJoinSubstitution(
            [
                FindPackageShare("mr_het_coord_ros"),
                "cfg",
                PythonExpression(
                    [
                        "'config_' + '",
                        LaunchConfiguration("num_agents"),
                        "' + 'agents.yml'",
                    ]
                ),
            ]
        ),
    ]

    # Create the node with the parameters loaded from the YAML file.
    mr_het_vrp_node = Node(
        package="mr_het_coord_ros",
        executable="mr_het_vrp_node",
        name="mr_het_vrp_node",
        output="screen",
        parameters=mr_coord_node_parameters,
        remappings=[
            ("/grid_map_in", "/grid_map_out"),
        ],
        # prefix=['gdbserver localhost:3000']
    )

    mr_het_map_node = Node(
        package="mr_het_coord_ros",
        executable="mr_het_map_node",
        name="mr_het_map_node",
        output="screen",
        parameters=mr_coord_node_parameters,
    )

    delayed_map_node = TimerAction(period=1.0, actions=[mr_het_map_node])

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("mr_het_coord_ros"), "cfg", "rviz.rviz"]
            ),
        ],
    )

    return LaunchDescription(
        [map_arg, num_agents_arg, rviz_node, mr_het_vrp_node, delayed_map_node]
    )
