#ifndef MR_HET_COORD_PRM_HPP
#define MR_HET_COORD_PRM_HPP

#include "mr_het_coord/map/map.hpp"
#include "mr_het_coord/planners/base_planner.hpp"
#include "mr_het_coord/planners/graph/astar_graph.hpp"
#include "mr_het_coord/planners/graph/graph_base.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

class PRM : public BasePlanner
{
public:
  PRM(GridMap &map, std::vector<std::vector<float>> &tasks, float step_size,
      float search_radius, std::mt19937& rng)
      : BasePlanner(),
        map_(map),
        step_size_(step_size),
        goal_th_(step_size * 1.2f),
        visibility_step_(step_size / 10.0f),
        search_radius_(search_radius),
        minXY_(map.getMinXY()),
        maxXY_(map.getMaxXY()),
        rng_(rng)
  {
    graph_ = buildGraphFromTasks_(map, tasks);
  }

  /**
   * Search for a path between start and goal, using A* on the PRM graph.
   *
   * @param start The starting position.
   * @param goal The goal position.
   * @param dist_th The maximum distance between two nodes in the path.
   * @return The path from start to goal, as a vector of 2D points.
   */
  std::vector<std::vector<float>> searchPath(std::vector<float> start,
                                             std::vector<float> goal,
                                             float dist_th) const override;

  /**
   * Checks if a line segment between two points is visible in the map.
   *
   * This function determines the visibility of a line segment between
   * the start point (x1, y1) and the end point (x2, y2) within the map.
   * The visibility is determined using a small increment and the signed
   * distance field.
   *
   * @param x1 World x coordinate of the start point.
   * @param y1 World y coordinate of the start point.
   * @param x2 World x coordinate of the end point.
   * @param y2 World y coordinate of the end point.
   * @return True if the line segment is visible, false otherwise.
   */
  bool isVisible(float x1, float y1, float x2, float y2) const override
  {
    return map_.isVisible(x1, y1, x2, y2, map_.resolution/2, true);
  }

  /**
   * Retrieves all edges in the graph as pairs of node coordinates.
   *
   * This function iterates over all nodes in the graph and their respective
   * neighbors to construct a list of edges. Each edge is represented as a pair
   * of 2D coordinates, corresponding to the positions of the connected nodes.
   *
   * @return A vector of edges, where each edge is a vector containing two
   *         vectors of floats, representing the x and y coordinates of the
   *         start and end nodes.
   */
  std::vector<std::vector<std::vector<float>>> getEdges() const
  {
    std::vector<std::vector<std::vector<float>>> edges;
    for (auto node : graph_.getNodes())
    {
      for (auto neighbor : node->neighbors)
      {
        edges.push_back({{node->x, node->y}, {neighbor->x, neighbor->y}});
      }
    }
    return edges;
  }

private:
  BaseGraph graph_;
  GridMap const &map_;
  float step_size_;
  float goal_th_;
  float visibility_step_;
  float search_radius_;
  int max_samples_;
  std::mt19937 &rng_;

  std::vector<float> minXY_;
  std::vector<float> maxXY_;

  /**
   * Build a graph from a set of positions in the map.
   *
   * This function creates a BaseGraph from a set of positions. It adds all the
   * positions as nodes in the graph and then computes the neighbors of each
   * node using the computeNeighbors method.
   *
   * @param positions The positions to add as nodes in the graph
   * @return The graph with all the positions added as nodes
   */
  BaseGraph buildGraphFromPositions_(
      std::vector<std::vector<float>> const &positions);

  /**
   * Computes the neighbors for each node in the graph and adds edges
   * between nodes that are visible to each other.
   *
   * This function iterates over each node in the given graph and finds
   * its potential neighbors using the k-nearest neighbors approach. For each
   * neighbor, it checks if there is visibility between the current node and
   * the neighbor using the visibility function. If the nodes are
   * visible to each other, an edge is added between them in the graph.
   *
   * @param graph The graph in which to compute neighbors and add edges
   */
  void computeNeighbors_(BaseGraph &graph, bool only_add = false, int k = 5);

  /**
   * Builds a graph from a map by sampling positions in the map and
   * building a graph from these positions.
   *
   * This function samples max_samples_ positions from the given map
   * and then calls buildGraphFromPositions_ to build a graph from
   * these positions. The resulting graph is then returned.
   *
   * @param map The map to use for sampling positions
   * @param max_samples The maximum number of positions to sample
   * @param rng The random number generator to use for sampling
   * @return The graph built from the sampled positions
   */
  BaseGraph buildGraphFromMap_(GridMap const &map, int max_samples);

  void componentsDist(std::vector<ConnectedComponent> const &connected, std::vector<std::vector<float>>&component_pairs);

  /**
   * Builds a graph from a set of tasks in the map.
   *
   * It first builds the graph using the positions in the tasks.
   * Then, it checks if the graph is connected and if not, it tries
   * to connect the components by finding paths between them.
   * Finally, it densifies the graph by finding new paths between
   * nodes that have less than 5 neighbors.
   *
   * @param map The map where the graph is built
   * @param tasks The tasks that are used to build the graph
   * @return The built graph
   */
  BaseGraph buildGraphFromTasks_(GridMap const &map,
                                 std::vector<std::vector<float>> const &tasks);

  std::vector<float> randomSample_(GridMap const &map, std::mt19937 &rng) const;

  float euclideanDist(std::vector<float> const &p1,
                      std::vector<float> const &p2) const
  {
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    return std::sqrt(dx * dx + dy * dy);
  }
};

#endif