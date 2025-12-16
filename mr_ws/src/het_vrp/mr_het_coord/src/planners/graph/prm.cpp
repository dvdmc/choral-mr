#include "mr_het_coord/planners/graph/prm.hpp"

#include <chrono>

#include "mr_het_coord/planners/map/rrt_star.hpp"

std::vector<float> PRM::randomSample_(GridMap const& map,
                                      std::mt19937& rng) const {
  std::uniform_real_distribution<float> x_dist(minXY_[0], maxXY_[0]);
  std::uniform_real_distribution<float> y_dist(minXY_[1], maxXY_[1]);
  float x = x_dist(rng);
  float y = y_dist(rng);
  while (!map.isValid(x, y, true)) {
    x = x_dist(rng);
    y = y_dist(rng);
  }
  return {x, y};
}

BaseGraph PRM::buildGraphFromPositions_(
    std::vector<std::vector<float>> const& positions) {
  BaseGraph graph;

  int id = 0;
  for (auto position : positions) {
    BaseGraphNode::Ptr node =
        std::make_shared<BaseGraphNode>(id, position[0], position[1]);
    graph.addNode(node);
    id++;
  }
  computeNeighbors_(graph, false, 5);

  return graph;
}

void PRM::computeNeighbors_(BaseGraph& graph, bool only_add, int k) {
  for (auto node : graph.getNodes()) {
    std::vector<BaseGraphNode::Ptr> neighbors;
    if (only_add && node->neighbors.size() < k) {
      neighbors =
          graph.getKNeighbors(node, k - node->neighbors.size(), true, true);
    } else {
      neighbors = graph.getKNeighbors(node, k, true, false);
    }
    // std::cout << "Node " << node->id << " has " << neighbors.size() << "
    // neighbors" << std::endl;
    for (auto neighbor : neighbors) {
      if (!node->neighbors.count(neighbor) &&
          isVisible(node->x, node->y, neighbor->x, neighbor->y)) {
        graph.addEdge(node, neighbor);
      }
    }
  }
}

BaseGraph PRM::buildGraphFromMap_(GridMap const& map, int max_samples) {
  // Sample in the map
  std::vector<std::vector<float>> positions;
  for (int i = 0; i < max_samples; i++) {
    positions.push_back(randomSample_(map, rng_));
  }

  // Build the graph
  return buildGraphFromPositions_(positions);
}

void PRM::componentsDist(std::vector<ConnectedComponent> const& connected,
                         std::vector<std::vector<float>>& component_dists) {
  for (int i = 0; i < connected.size(); i++) {
    for (int j = i + 1; j < connected.size(); j++) {
      float min_distance = std::numeric_limits<float>::max();
      for (auto& node1 : connected[i].nodes) {
        for (auto& node2 : connected[j].nodes) {
          float dist =
              euclideanDist({node1->x, node1->y}, {node2->x, node2->y});
          if (dist < min_distance) min_distance = dist;
        }
      }
      component_dists[i][j] = min_distance;
      component_dists[j][i] = min_distance;  // symmetric
    }
  }
}

BaseGraph PRM::buildGraphFromTasks_(
    GridMap const& map, std::vector<std::vector<float>> const& tasks) {
  auto start = std::chrono::high_resolution_clock::now();
  // Print map
  // std::cout << "Map: " << std::endl;
  // map.printMap();
  // map.printSdfMap();
  std::vector<std::vector<float>> positions;
  // Add the tasks first
  for (int i = 0; i < tasks.size(); i++) {
    positions.push_back(tasks[i]);
  }

  // Build the basic graph
  BaseGraph graph = buildGraphFromPositions_(positions);

  // std::cout << "Time taken in building from positions: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  std::chrono::high_resolution_clock::now() - start)
  //                  .count()
  //           << " ms" << std::endl;

  // Compute missing paths
  std::cout << "Checking connected components" << std::endl;
  auto start2 = std::chrono::high_resolution_clock::now();
  std::vector<ConnectedComponent> connected = graph.findConnectedComponents();
  // std::cout << "Time taken in finding components: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  std::chrono::high_resolution_clock::now() - start2)
  //                  .count()
  //           << " ms" << std::endl;
  std::cout << "Found " << connected.size() << " components." << std::endl;

  /////////// CONNECT DISCONNECTED COMPONENTS
  RRTStar rrt(map, step_size_, 10000, 5.0f, rng_, false);
  auto start3 = std::chrono::high_resolution_clock::now();

  if (connected.size() > 1) {
    // Order the components from smaller to larger size
    std::sort(connected.begin(), connected.end(),
              [](ConnectedComponent const& a, ConnectedComponent const& b) {
                return a.nodes.size() < b.nodes.size();
              });

    for (size_t i = 0; i < connected.size(); ++i) {
      std::cout << "Component " << connected[i].id << " has "
                << connected[i].nodes.size() << " nodes" << std::endl;
    }

    // Calculate the distance between components
    std::vector<std::vector<float>> component_dists(
        connected.size(),
        std::vector<float>(connected.size(),
                           std::numeric_limits<float>::max()));
    componentsDist(connected, component_dists);

    // Try to connect nearest components
    std::cout << "Connecting components" << std::endl;
    for (size_t i = 0; i < connected.size(); ++i) {
      // Temporary sorted copy for distance-based ordering
      std::vector<ConnectedComponent> tmp_connected = connected;

      // Sort tmp_connected based on distance from current component
      std::sort(tmp_connected.begin(), tmp_connected.end(),
                [&component_dists, i, &connected](ConnectedComponent const& a,
                                                  ConnectedComponent const& b) {
                  // Map component id -> index
                  auto it_a = std::find_if(connected.begin(), connected.end(),
                                           [&a](const ConnectedComponent& c) {
                                             return c.id == a.id;
                                           });
                  auto it_b = std::find_if(connected.begin(), connected.end(),
                                           [&b](const ConnectedComponent& c) {
                                             return c.id == b.id;
                                           });
                  size_t idx_a = std::distance(connected.begin(), it_a);
                  size_t idx_b = std::distance(connected.begin(), it_b);
                  return component_dists[i][idx_a] < component_dists[i][idx_b];
                });

      // Attempt to connect to all smaller-index components
      for (size_t j = 0; j < i; ++j) {
        bool is_connected = false;
        std::cout << "Trying to connect components " << connected[i].id
                  << " and " << connected[j].id << std::endl;
        std::cout << "Safe: " << map_.FREE_THRESH << std::endl;
        auto potential = graph.getKPotentialConnections(
            connected[i].nodes, tmp_connected[j].nodes, 10);

        for (auto& pair : potential) {
          auto path =
              rrt.searchPath({pair.first->x, pair.first->y},
                             {pair.second->x, pair.second->y}, goal_th_);

          if (!path.empty()) {
            std::cout << "Found path!" << std::endl;
            is_connected = true;

            // Add intermediate nodes
            if (path.size() > 2) {
              auto prev_node = pair.first;
              for (size_t k = 1; k < path.size() - 1; ++k) {
                auto new_node = std::make_shared<BaseGraphNode>(
                    graph.size(), path[k][0], path[k][1]);
                graph.addNode(new_node);
                graph.addEdge(prev_node, new_node);
                prev_node = new_node;
              }
              // Connect last node
              graph.addEdge(prev_node, pair.second);
            } else {
              // Directly connect nodes
              graph.addEdge(pair.first, pair.second);
            }
            break;  // stop after first successful path
          }
        }

        if (is_connected) break;  // move to next component
      }
    }

    // Recompute neighbors
    computeNeighbors_(graph, true);

    // Check if fully connected
    std::cout << "Re-checking connected components" << std::endl;
    connected = graph.findConnectedComponents();
    if (connected.size() > 1) {
      std::cout << "Disconnected components!" << std::endl;
      std::cout << "Found components: " << connected.size() << std::endl;
      for (size_t i = 0; i < connected.size(); ++i) {
        std::cout << "Component " << connected[i].id << " has "
                  << connected[i].nodes.size() << " nodes" << std::endl;
      }
      throw std::runtime_error("Failed to connect components");
    }
  }

  // std::cout << "Time taken in connecting disconnected: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  std::chrono::high_resolution_clock::now() - start3)
  //                  .count()
  //           << " ms" << std::endl;
  /////////// FINISH CONNECTING DISCONNECTED COMPONENTS

  // NOTE: Densifying proved to not improve much in the final version of the PRM construction.
  // It might improve results in more complex environemnts
  std::cout << "Densifying graph" << std::endl; auto start4 =
  std::chrono::high_resolution_clock::now();
  //Densify, if any node has less than 5 neighbors, find paths to others
  auto nodes = graph.getNodes();
  for (auto node : nodes) {
    if (node->neighbors.size() < std::min(5, (int)graph.size() - 1)) {
      // std::cout << "Node " << node->id << " has " <<
      // node->neighbors.size() << " neighbors. Densifying" << std::endl;
      std::vector<BaseGraphNode::Ptr> potential =
          graph.getKNeighbors(node, 5, true, true);
      for (auto const& neighbor : potential) {
        std::cout << "Searching path for densification between node "
                  << node->id << " and " << neighbor->id << std::endl;
        std::vector<std::vector<float>> path = rrt.searchPath(
            {node->x, node->y}, {neighbor->x, neighbor->y}, goal_th_);
        if (path.size() > 0) {
          for (int i = 1; i < path.size() - 1; i++) {
            BaseGraphNode::Ptr new_node = std::make_shared<BaseGraphNode>(
                graph.size(), path[i][0], path[i][1]);
            graph.addNode(new_node);
          }
          // break; // Optionally, include just one
        }
        computeNeighbors_(graph);
      }
    }
  }
  computeNeighbors_(graph);
    std::cout << "Time taken in densifying: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - start4)
                   .count()
            << " ms" << std::endl;

  std::cout << "############## Total time taken in building PRM: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count()
            << " ms" << std::endl;
  std::cout << "Finish graph" << std::endl;
  return graph;
}

std::vector<std::vector<float>> PRM::searchPath(std::vector<float> start,
                                                std::vector<float> goal,
                                                float dist_th) const {
  return AStarGraph(graph_, map_).searchPath(start, goal, dist_th);
}
