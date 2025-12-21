cmake_minimum_required(VERSION 3.18)

project(mr_het_coord)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-O3 -Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/or-tools")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(ortools REQUIRED)

include_directories(
  include
  ${GTEST_INCLUDE_DIRS}
)

add_library(MRHetCoordlib
src/planners/astar_base.cpp
src/planners/map/rrt_star.cpp
src/planners/graph/graph_base.cpp
src/planners/graph/astar_graph.cpp
src/planners/graph/prm.cpp
src/vrpsolver.cpp
)

target_include_directories(MRHetCoordlib
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

target_link_libraries(MRHetCoordlib
  ortools::ortools
)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS MRHetCoordlib ortools
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

find_package(ament_cmake_gtest)
ament_add_gtest(some_test test/test_map.cpp)

ament_package()