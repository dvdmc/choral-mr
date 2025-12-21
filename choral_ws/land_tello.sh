#!/bin/bash

# Default value: drone0
drone=${drone:-drone0}

sleep 0.5
ros2 service call /$drone/set_offboard_mode std_srvs/srv/SetBool "data: true"

sleep 0.5
ros2 service call /$drone/platform/state_machine_event as2_msgs/srv/SetPlatformStateMachineEvent "event:
  event: 4"

sleep 0.5
ros2 service call /$drone/platform_land std_srvs/srv/SetBool "data: true"