#!/bin/bash
set -e

# Source ROS 2 setup if available
if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    source "/opt/ros/$ROS_DISTRO/setup.bash"
fi

# Add custom workspace setup if available


# Execute the passed command
exec "$@"
