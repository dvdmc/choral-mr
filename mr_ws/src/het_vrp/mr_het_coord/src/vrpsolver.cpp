#include "mr_het_coord/vrpsolver.hpp"

#include <chrono>

#include "mr_het_coord/utils/thread_pool.hpp"

void VRPSolver::distanceFromPathPlanner(
    std::vector<std::vector<float>> const& positions, BasePlanner& path_planner,
    std::vector<std::vector<int64_t>>& distance_matrix,
    std::vector<std::vector<std::vector<std::vector<float>>>>& paths) const {
  distance_matrix = std::vector<std::vector<int64_t>>(
      positions.size(), std::vector<int64_t>(positions.size(), 0));

  paths = std::vector<std::vector<std::vector<std::vector<float>>>>(
      positions.size(),
      std::vector<std::vector<std::vector<float>>>(
          positions.size(), std::vector<std::vector<float>>(0)));

  std::cout << "Start distances" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  int total = positions.size() * positions.size();
  for (size_t i = 0; i < distance_matrix.size(); ++i) {
    for (size_t j = 0; j <= i; ++j) {
      std::vector<float> position1 = positions[i];
      std::vector<float> position2 = positions[j];

      if (i == j) {
        paths[i][j] = {position1, position2};
        paths[j][i] = {position2, position1};
        distance_matrix[i][j] = 0;
        distance_matrix[j][i] = 0;
        continue;
      }

      // NOTE: Can return an empty path, resulting in -1 distance
      paths[i][j] = path_planner.searchPath(position1, position2, 0.2f);
      paths[j][i] = path_planner.invertPath(paths[i][j]);

      float path_dist = path_planner.computeDistance(paths[i][j]);
      if (path_dist == -1) {
        path_dist = int64_t(IMPOSIBLE_COST);
      }
      distance_matrix[i][j] = int(path_dist * DISTANCE_SCALING);
      distance_matrix[j][i] = distance_matrix[i][j];
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "############## Time taken on computing distances: "
            << duration.count() << " ms" << std::endl;
  makeAsymmetric_(distance_matrix);
}

void VRPSolver::distanceFromPaths(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& paths,
    const BasePlanner& path_planner,
    std::vector<std::vector<int64_t>>& distance_matrix) const {
  distance_matrix = std::vector<std::vector<int64_t>>(
      paths.size(), std::vector<int64_t>(paths.size(), 0));

  std::cout << "Start distances" << std::endl;
  int total = paths.size() * paths.size();
  for (size_t i = 0; i < distance_matrix.size(); ++i) {
    for (size_t j = 0; j <= i; ++j) {
      float path_dist = path_planner.computeDistance(paths[i][j]);
      if (path_dist == -1) {
        std::cout << "Not found path between " << i << ", and " << j << "!"
                  << std::endl;
        path_dist = int64_t(IMPOSIBLE_COST);
      }
      distance_matrix[i][j] = int(path_dist * DISTANCE_SCALING);
      distance_matrix[j][i] = distance_matrix[i][j];
    }
  }
  std::cout << "Finish distances" << std::endl;
  // Print distance matrix
  for (size_t i = 0; i < distance_matrix.size(); ++i) {
    for (size_t j = 0; j < distance_matrix.size(); ++j) {
      std::cout << "From " << i << " to " << j << ": " << distance_matrix[i][j]
                << std::endl;
    }
    std::cout << std::endl;
  }
  makeAsymmetric_(distance_matrix);
}

void VRPSolver::uniformCostMatrices(
    std::vector<std::vector<int64_t>> const& distance_matrix, int num_vehicles,
    std::vector<std::vector<std::vector<int64_t>>>& cost_matrices) const {
  cost_matrices = std::vector<std::vector<std::vector<int64_t>>>(
      num_vehicles, std::vector<std::vector<int64_t>>(
                        distance_matrix.size(),
                        std::vector<int64_t>(distance_matrix.size(), 0)));
  // Consider untraversable paths
  for (size_t i = 0; i < distance_matrix.size(); ++i) {
    for (size_t j = 0; j < distance_matrix.size(); ++j) {
      if (distance_matrix[i][j] >= IMPOSIBLE_COST * DISTANCE_SCALING) {
        for (size_t k = 0; k < num_vehicles; ++k) {
          cost_matrices[k][i][j] = IMPOSIBLE_COST * DISTANCE_SCALING;
        }
      }
    }
  }
  makeAsymmetric_(cost_matrices);
}

void VRPSolver::hetPlatformCostMatrices(
    GridMap const& grid_map, BasePlanner const& path_planner, int num_vehicles,
    std::vector<std::vector<std::vector<std::vector<float>>>> const& paths,
    std::vector<AgentType> const& agent_types,
    std::vector<std::vector<std::vector<int64_t>>>& cost_matrices,
    float lambda_good_trav, float lambda_bad_trav, float gamma_collision,
    float d_05_collision) const {
  auto start = std::chrono::high_resolution_clock::now();

  // We use the paths as the distance matrix
  cost_matrices = std::vector<std::vector<std::vector<int64_t>>>(
      num_vehicles,
      std::vector<std::vector<int64_t>>(
          paths.size(), std::vector<int64_t>(paths[0].size(), 0)));

  std::vector<std::vector<int64_t>> aerial_cost_matrix(
      paths.size(), std::vector<int64_t>(paths[0].size(), 0));
  std::vector<std::vector<int64_t>> ground_cost_matrix(
      paths.size(), std::vector<int64_t>(paths[0].size(), 0));

  for (size_t i = 0; i < paths.size(); ++i)  // V is the number of nodes
  {
    for (size_t j = 0; j < i; ++j)  // V is the number of nodes
    {
      // Consider untraversable paths
      if(paths[i][j].empty()) {
        aerial_cost_matrix[i][j] = IMPOSIBLE_COST * DISTANCE_SCALING;
        aerial_cost_matrix[j][i] = IMPOSIBLE_COST * DISTANCE_SCALING;
        ground_cost_matrix[i][j] = IMPOSIBLE_COST * DISTANCE_SCALING;
        ground_cost_matrix[j][i] = IMPOSIBLE_COST * DISTANCE_SCALING;
        continue;
      }

      float aerial_safety_cost = path_planner.computeSafetyCost(
                            grid_map, paths[i][j], AgentType::AERIAL,
                            lambda_good_trav, lambda_bad_trav,
                            gamma_collision, d_05_collision) *
                        DISTANCE_SCALING;

      aerial_cost_matrix[i][j] = aerial_safety_cost;
      aerial_cost_matrix[j][i] = aerial_cost_matrix[i][j];

      float ground_safety_cost = path_planner.computeSafetyCost(
                                   grid_map, paths[i][j], AgentType::GROUND,
                                   lambda_good_trav, lambda_bad_trav,
                                   gamma_collision, d_05_collision) *
                               DISTANCE_SCALING;

      ground_cost_matrix[i][j] = ground_safety_cost;
      ground_cost_matrix[j][i] = ground_cost_matrix[i][j];
    }
  }

  for (size_t k = 0; k < num_vehicles; ++k) {
    if (agent_types[k] == AgentType::AERIAL) {
      cost_matrices[k] = aerial_cost_matrix;
    } else {
      cost_matrices[k] = ground_cost_matrix;
    }
  }

  makeAsymmetric_(cost_matrices);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken on computing het costs: " << duration.count()
            << " ms" << std::endl;
}

void VRPSolver::addCostHeterogeneous_(
    RoutingModel& routing, RoutingIndexManager& manager,
    std::vector<std::vector<int64_t>> const& distance_matrix,
    std::vector<std::vector<std::vector<int64_t>>> const& cost_matrices,
    float cost_scaling, std::vector<float> velocities,
    std::vector<int>& transit_callback_indeces) const {

  for (size_t vehicle_id = 0; vehicle_id < cost_matrices.size(); ++vehicle_id) {
    int64_t const transit_callback_index_vehicle =
        routing.RegisterTransitCallback(
            [&cost_matrices, &distance_matrix, &manager, vehicle_id,
             cost_scaling, &velocities](int64_t const from_index,
                                        int64_t const to_index) -> int64_t {
              int const from_node = manager.IndexToNode(from_index).value();
              int const to_node = manager.IndexToNode(to_index).value();
              // CAUTION: This might overflow and the max possible cost is not
              // enforced anymore
              return int64_t(double(distance_matrix[from_node][to_node]) /
                             velocities[vehicle_id]) +
                     int64_t(cost_scaling *
                         double(cost_matrices[vehicle_id][from_node][to_node]));
            });

    transit_callback_indeces.push_back(
        transit_callback_index_vehicle);  // For max span

    routing.SetArcCostEvaluatorOfVehicle(transit_callback_index_vehicle,
                                         vehicle_id);
  }
}

void VRPSolver::addMaxSpanConstraint_(
    RoutingModel& routing, int64_t median, int64_t min_cost, int num_nodes,
    int num_vehicles, std::vector<int> transit_callback_indeces) const {
  std::cout << "Median distance: " << median << std::endl;
  std::cout << "Median (risk) cost: " << min_cost << std::endl;
  // Minimize each vehicles' global span
  int64_t est_route_cost = (median * (num_nodes) / (num_vehicles)) +
                           (min_cost * (num_nodes) / (num_vehicles)) *
                               0.001;  // Ideally no added cost
  std::cout << "Median route cost: " << est_route_cost << std::endl;

  routing.AddDimensionWithVehicleTransits(
      transit_callback_indeces, int64_t{0}, est_route_cost,
      true,         // start cumul to zero
      "Distance");  // Format the name of the vehicle d1, d2, ...
  routing.GetMutableDimension("Distance")->SetGlobalSpanCostCoefficient(100);
}

RoutingSearchParameters VRPSolver::addInitialHeuristic_(
    RoutingIndexManager& manager, RoutingModel& routing,
    int solver_seconds) const {
  // Setting first solution heuristic.
  RoutingSearchParameters search_parameters = DefaultRoutingSearchParameters();
  search_parameters.set_first_solution_strategy(
      FirstSolutionStrategy::PATH_CHEAPEST_ARC);
  search_parameters.set_local_search_metaheuristic(
      LocalSearchMetaheuristic::GUIDED_LOCAL_SEARCH);
  search_parameters.mutable_time_limit()->set_seconds(solver_seconds);

  return search_parameters;
}

int64_t VRPSolver::getMaxDistance_(
    std::vector<std::vector<int64_t>> const& distance_matrix) const {
  int64_t max_distance = 0;
  for (size_t i = 0; i < distance_matrix.size(); ++i) {
    for (size_t j = 0; j < distance_matrix.size(); ++j) {
      if (distance_matrix[i][j] > max_distance &&
          distance_matrix[i][j] != IMPOSIBLE_COST * DISTANCE_SCALING) {
        max_distance = distance_matrix[i][j];
      }
    }
  }
  return max_distance;
}

int64_t VRPSolver::getMaxCost_(
    std::vector<std::vector<std::vector<int64_t>>> const& cost_matrices) const {
  int64_t max_cost = 0;
  for (size_t k = 0; k < cost_matrices.size(); ++k) {
    for (size_t i = 0; i < cost_matrices[k].size(); ++i) {
      for (size_t j = 0; j < cost_matrices[k].size(); ++j) {
        if (cost_matrices[k][i][j] > max_cost &&
            cost_matrices[k][i][j] != IMPOSIBLE_COST * DISTANCE_SCALING) {
          max_cost = cost_matrices[k][i][j];
        }
      }
    }
  }
  return max_cost;
}

std::vector<int64_t> VRPSolver::getMedianCost_(
    std::vector<std::vector<std::vector<int64_t>>> const& cost_matrices) const {
  std::vector<int64_t> median_costs;
  for (size_t k = 0; k < cost_matrices.size(); ++k) {
    std::vector<int64_t> costs;
    for (size_t i = 0; i < cost_matrices[k].size(); ++i) {
      for (size_t j = 0; j < cost_matrices[k].size(); ++j) {
        if (cost_matrices[k][i][j] != IMPOSIBLE_COST * DISTANCE_SCALING) {
          costs.push_back(cost_matrices[k][i][j]);
        }
      }
    }
    if (costs.empty()) {
      median_costs.push_back(
          0);  // Or handle as an error, depending on requirements
      continue;
    }
    std::sort(costs.begin(), costs.end());
    size_t size = costs.size();
    if (size % 2 == 0) {
      // Even number of elements
      median_costs.push_back((costs[size / 2 - 1] + costs[size / 2]) / 2);
    } else {
      // Odd number of elements
      median_costs.push_back(costs[size / 2]);
    }
  }
  return median_costs;
}

int64_t VRPSolver::getMedianDistance_(
    std::vector<std::vector<int64_t>> const& distance_matrix) const {
  std::vector<int64_t> distances;
  for (size_t i = 0; i < distance_matrix.size(); ++i) {
    for (size_t j = 0; j < distance_matrix.size(); ++j) {
      if (distance_matrix[i][j] != IMPOSIBLE_COST * DISTANCE_SCALING) {
        distances.push_back(distance_matrix[i][j]);
      }
    }
  }
  if (distances.empty()) {
    return 0;  // Or handle as an error
  }
  std::sort(distances.begin(), distances.end());
  size_t size = distances.size();
  if (size % 2 == 0) {
    // Even number of elements
    return (distances[size / 2 - 1] + distances[size / 2]) / 2;
  } else {
    // Odd number of elements
    return distances[size / 2];
  }
}

void VRPSolver::normalizeDistanceMatrix_(
    std::vector<std::vector<int64_t>>& distance_matrix) const {
  int64_t max_distance = getMaxDistance_(distance_matrix);
  if (max_distance == 0) {
    return;
  }
  double scaling = (double)max_distance / (double)DISTANCE_SCALING;
  std::cout << "Scaling distance: " << scaling << std::endl;
  for (size_t i = 0; i < distance_matrix.size(); ++i) {
    for (size_t j = 0; j < distance_matrix.size(); ++j) {
      if (distance_matrix[i][j] != IMPOSIBLE_COST * DISTANCE_SCALING) {
        distance_matrix[i][j] = (double)distance_matrix[i][j] / scaling;
      }
    }
  }
}

void VRPSolver::normalizeCostMatrices_(
    std::vector<std::vector<std::vector<int64_t>>>& cost_matrices) const {
  int64_t max_cost = getMaxCost_(cost_matrices);
  if (max_cost == 0) {
    return;
  }
  double scaling =
      (double)max_cost /
      (double)DISTANCE_SCALING;  // Get the scaling without the scaling part
  std::cout << "Scaling cost: " << scaling << std::endl;
  for (size_t k = 0; k < cost_matrices.size(); ++k) {
    for (size_t i = 0; i < cost_matrices[k].size(); ++i) {
      for (size_t j = 0; j < cost_matrices[k].size(); ++j) {
        if (cost_matrices[k][i][j] != IMPOSIBLE_COST * DISTANCE_SCALING) {
          cost_matrices[k][i][j] = (double)cost_matrices[k][i][j] / scaling;
        }
      }
    }
  }
}

std::vector<std::vector<int64_t>> VRPSolver::solveWithMaxSpan(
    std::vector<std::vector<int64_t>> distance_matrix,
    std::vector<std::vector<std::vector<int64_t>>> cost_matrices,
    float cost_scaling, std::vector<float> velocities, int num_vehicles,
    int depot_idx, int solver_seconds,
    std::vector<int64_t>& resulting_route_costs) const {
  // Init the solver
  const RoutingIndexManager::NodeIndex depot{depot_idx};
  RoutingIndexManager manager(distance_matrix.size(), num_vehicles, depot);
  RoutingModel routing(manager);

  // Normalize matrix so values are comparable
  normalizeDistanceMatrix_(distance_matrix);
  normalizeCostMatrices_(cost_matrices);

  // Get median distance for adding a capacity in MaxSpan
  int64_t median_dist = getMedianDistance_(distance_matrix);

  int64_t min_time_dist = median_dist;
  for (size_t i = 1; i < velocities.size(); ++i) {
    int64_t time = double(median_dist) / velocities[i];
    if (velocities[i] < min_time_dist) {
      min_time_dist = time;
    }
  }

  std::vector<int64_t> median_costs = getMedianCost_(cost_matrices);

  int64_t min_median_cost = median_costs[0];
  for (size_t i = 1; i < median_costs.size(); ++i) {
    if (median_costs[i] < min_median_cost) {
      min_median_cost = median_costs[i];
    }
  }

  std::vector<int>
      transit_callback_indeces;  // Keep this for adding constraints later
  addCostHeterogeneous_(routing, manager, distance_matrix, cost_matrices,
                        cost_scaling, velocities, transit_callback_indeces);

  addMaxSpanConstraint_(routing, min_time_dist, min_median_cost,
                        distance_matrix.size(), num_vehicles,
                        transit_callback_indeces);

  // Setting first solution heuristic.
  RoutingSearchParameters search_parameters =
      addInitialHeuristic_(manager, routing, solver_seconds);

  // Solve!
  Assignment const* solution = routing.SolveWithParameters(search_parameters);
  if (solution == nullptr) {
    std::cout << "No solution found for VRP!" << std::endl;
    return {};
  }
  // Print solution on console.
  printSolution_(manager, routing, *solution);
  // Get route costs for the metrics
  getRouteCosts_(manager, routing, *solution, resulting_route_costs);

  return getSolution_(manager, routing, *solution);
}

std::vector<std::vector<int64_t>> VRPSolver::getSolution_(
    RoutingIndexManager const& manager, RoutingModel const& routing,
    Assignment const& solution) const {
  int num_vehicles = routing.vehicles();
  std::vector<std::vector<int64_t>> routes(num_vehicles);
  for (size_t vehicle_id = 0; vehicle_id < num_vehicles; ++vehicle_id) {
    int64_t index = routing.Start(vehicle_id);
    while (!routing.IsEnd(index)) {
      int const node_index = manager.IndexToNode(index).value();
      index = solution.Value(routing.NextVar(index));
      routes[vehicle_id].push_back(node_index);
    }
  }
  return routes;
}

void VRPSolver::printSolution_(RoutingIndexManager const& manager,
                               RoutingModel const& routing,
                               Assignment const& solution) const {
  int64_t total_cost = 0;
  // int64_t total_load = 0;
  for (size_t vehicle_id = 0; vehicle_id < routing.vehicles(); ++vehicle_id) {
    int64_t index = routing.Start(vehicle_id);
    LOG(INFO) << "Route for Vehicle " << vehicle_id << ":";
    int64_t route_cost = 0;
    // int64_t route_load = 0;
    std::stringstream route;
    while (!routing.IsEnd(index)) {
      int const node_index = manager.IndexToNode(index).value();
      // route_load += data.demands[node_index];
      route << node_index << " -> ";
      int64_t const previous_index = index;
      index = solution.Value(routing.NextVar(index));
      route_cost = route_cost + routing.GetArcCostForVehicle(
                                    previous_index, index, int64_t{vehicle_id});
    }
    LOG(INFO) << route.str() << manager.IndexToNode(index).value();
    LOG(INFO) << "Cost of the route: " << float(route_cost) << "m";
    total_cost += route_cost;
  }
  LOG(INFO) << "Total cost of all routes: " << total_cost << "m";
  LOG(INFO) << "";
  LOG(INFO) << "Problem solved in " << routing.solver()->wall_time() << "ms";
}

std::vector<int64_t> VRPSolver::getRouteCosts_(
    RoutingIndexManager const& manager, RoutingModel const& routing,
    Assignment const& solution, std::vector<int64_t>& route_costs) const {
  int num_vehicles = routing.vehicles();
  route_costs = std::vector<int64_t>(num_vehicles, 0);
  for (size_t vehicle_id = 0; vehicle_id < routing.vehicles(); ++vehicle_id) {
    int64_t index = routing.Start(vehicle_id);
    int64_t route_distance = 0;
    while (!routing.IsEnd(index)) {
      int64_t const previous_index = index;
      index = solution.Value(routing.NextVar(index));
      route_distance =
          route_distance + routing.GetArcCostForVehicle(previous_index, index,
                                                        int64_t{vehicle_id});
    }
    route_costs[vehicle_id] = route_distance;
  }
  return route_costs;
}
