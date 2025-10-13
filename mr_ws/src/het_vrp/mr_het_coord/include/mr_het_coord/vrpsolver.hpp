#ifndef MR_HET_COORD_VRPSOLVER_HPP
#define MR_HET_COORD_VRPSOLVER_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "mr_het_coord/planners/base_planner.hpp"
#include "ortools/constraint_solver/routing.h"
#include "ortools/constraint_solver/routing_enums.pb.h"
#include "ortools/constraint_solver/routing_index_manager.h"
#include "ortools/constraint_solver/routing_parameters.h"

using namespace operations_research;

class VRPSolver {
 public:
  VRPSolver() {};

  /**
   * Computes the distance matrix and paths between all positions using the
   * provided path planner.
   *
   * For each pair of positions, this function calculates the path using the
   * path planner and computes the path distance. If the path cannot be found,
   * the function approximates the distance using the Euclidean distance and
   * applies a scaling factor. The distance matrix is made asymmetric by setting
   * the arcs to the depot to 0.
   *
   * @param positions The list of positions as (x, y) coordinates.
   * @param path_planner The path planner used to compute paths and distances.
   * @param[out] distance_matrix The computed distance matrix.
   * @param[out] paths A 4D vector representing paths between each pair of
   * positions.
   */
  void distanceFromPathPlanner(
      std::vector<std::vector<float>> const& positions,
      BasePlanner& path_planner,
      std::vector<std::vector<int64_t>>& distance_matrix,
      std::vector<std::vector<std::vector<std::vector<float>>>>& paths) const;

  void distanceFromPaths(
      const std::vector<std::vector<std::vector<std::vector<float>>>>& paths,
      const BasePlanner& path_planner,
      std::vector<std::vector<int64_t>>& distance_matrix) const;

  void uniformCostMatrices(
      std::vector<std::vector<int64_t>> const& distance_matrix,
      int num_vehicles,
      std::vector<std::vector<std::vector<int64_t>>>& cost_matrices) const;

  void hetPlatformCostMatrices(
      GridMap const& grid_map, BasePlanner const& path_planner,
      int num_vehicles,
      std::vector<std::vector<std::vector<std::vector<float>>>> const& paths,
      std::vector<AgentType> const& agent_types,
      std::vector<std::vector<std::vector<int64_t>>>& cost_matrices,
      float lambda_good_trav, float lambda_bad_trav, float gamma_collision,
      float d_05_collision) const;

  std::vector<std::vector<int64_t>> solveWithMaxSpan(
      std::vector<std::vector<int64_t>> distance_matrix,
      std::vector<std::vector<std::vector<int64_t>>> cost_matrices,
      float cost_scaling, std::vector<float> velocities, int num_vehicles,
      int depot_idx, int solver_seconds,
      std::vector<int64_t>& resulting_route_distances) const;

  std::vector<std::vector<int64_t>> getSolution_(
      RoutingIndexManager const& manager, RoutingModel const& routing,
      Assignment const& solution) const;

  void printSolution_(RoutingIndexManager const& manager,
                      RoutingModel const& routing,
                      Assignment const& solution) const;

  std::vector<int64_t> getRouteCosts_(RoutingIndexManager const& manager,
                                      RoutingModel const& routing,
                                      Assignment const& solution,
                                      std::vector<int64_t>& route_costs) const;

 private:
  static constexpr int64_t DISTANCE_SCALING =
      1e4;  // For converting everything to integer
  static constexpr int64_t IMPOSIBLE_COST =
      1e10;  // As a prohibitive value that can also be converted to double

  void makeAsymmetric_(std::vector<std::vector<int64_t>> matrix) const {
    for (size_t i = 0; i < matrix.size(); ++i) {
      matrix[i][0] = 0;  // Going from any node to the virtual depot is free
    }
  }

  void makeAsymmetric_(
      std::vector<std::vector<std::vector<int64_t>>> matrix) const {
    for (size_t k = 0; k < matrix.size(); ++k) {
      for (size_t i = 0; i < matrix[k].size(); ++i) {
        matrix[k][i][0] =
            0;  // Going from any node to the virtual depot is free
      }
    }
  }

  int64_t getMedianDistance_(
      std::vector<std::vector<int64_t>> const& distance_matrix) const;

  std::vector<int64_t> getMedianCost_(
      std::vector<std::vector<std::vector<int64_t>>> const& cost_matrices)
      const;

  int64_t getMaxDistance_(
      std::vector<std::vector<int64_t>> const& distance_matrix) const;

  int64_t getMaxCost_(std::vector<std::vector<std::vector<int64_t>>> const&
                          cost_matrices) const;

  void normalizeDistanceMatrix_(
      std::vector<std::vector<int64_t>>& distance_matrix) const;

  void normalizeCostMatrices_(
      std::vector<std::vector<std::vector<int64_t>>>& cost_matrices) const;

  void addCostHeterogeneous_(
      RoutingModel& routing, RoutingIndexManager& manager,
      const std::vector<std::vector<int64_t>>& distance_matrix,
      const std::vector<std::vector<std::vector<int64_t>>>& cost_matrices,
      float cost_scaling, std::vector<float>& velocities,
      std::vector<int>& transit_callback_indeces) const;

  void addMaxSpanConstraint_(RoutingModel& routing, int64_t median_dist,
                             int64_t median_cost, int num_nodes, int num_vehicles,
                             std::vector<int> transit_callback_indeces) const;

  RoutingSearchParameters addInitialHeuristic_(RoutingIndexManager& manager,
                                               RoutingModel& routing,
                                               int solver_time) const;
};

#endif