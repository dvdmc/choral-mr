cmake_minimum_required(VERSION 3.18)

project(mr_het_coord_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-O3 -Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(mr_het_coord REQUIRED)

find_package(rosidl_default_generators REQUIRED)

find_package(rclcpp_action REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(as2_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
"msg/VrpSolution.msg"
DEPENDENCIES geometry_msgs
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

include_directories(
  include
)

add_executable(mr_het_vrp_node
  src/mr_het_vrp_node.cpp
)
target_link_libraries(mr_het_vrp_node "${cpp_typesupport_target}")
ament_target_dependencies(mr_het_vrp_node
  rclcpp
  tf2
  nav_msgs
  geometry_msgs
  visualization_msgs
  mr_het_coord
)

add_executable(mr_het_map_node
  src/mr_het_map_node.cpp
)
ament_target_dependencies(mr_het_map_node
  rclcpp
  nav_msgs
  mr_het_coord
)

add_executable(mr_het_planner_node
  src/mr_het_planner_node.cpp
)
target_link_libraries(mr_het_planner_node "${cpp_typesupport_target}")
ament_target_dependencies(mr_het_planner_node
  rclcpp
  tf2
  tf2_geometry_msgs
  geometry_msgs
  as2_msgs
  nav2_msgs
  mr_het_coord
  rclcpp_action
)

install(TARGETS 
    mr_het_vrp_node 
    mr_het_map_node
    mr_het_planner_node
    DESTINATION lib/${PROJECT_NAME})

# Install launch files.
install(DIRECTORY
launch
DESTINATION share/${PROJECT_NAME}/
)

# Install config files.
install(DIRECTORY
config
DESTINATION share/${PROJECT_NAME}/
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()