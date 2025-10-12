#ifndef MR_HET_COORD_ASTAR_MAP_HPP
#define MR_HET_COORD_ASTAR_MAP_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>

#include "mr_het_coord/map/map.hpp"
#include "mr_het_coord/planners/astar_base.hpp"
#include "mr_het_coord/planners/base_planner.hpp"
#include "mr_het_coord/planners/graph/graph_base.hpp"

// Extend the node class to include a pointer to the graph node
// This way we reuse the CompareNode but are able to expand from current to
// neighbors
struct AStarGraphNode : public AStarNode {
  using Ptr = std::shared_ptr<AStarGraphNode>;

  AStarGraphNode(float x, float y, float g = 0, float h = 0,
                 AStarNode::Ptr parent = nullptr,
                 BaseGraphNode::Ptr graph_node = nullptr)
      : AStarNode(x, y, g, h, parent), graph_node(graph_node) {}
  BaseGraphNode::Ptr graph_node;
};

class AStarGraph : public BasePlanner {
 public:
  /**
   * AStarGraph is an implementation of the A* algorithm on a graph structure.
   * The graph is used to generate a set of nodes and edges that can be
   * traversed. The A* algorithm is then used to find the shortest path from a
   * start node to a goal node in the graph.
   *
   * @param graph The graph structure to use for the A* algorithm
   * @param map The map to use for the A* algorithm
   */
  AStarGraph(BaseGraph const& graph, GridMap const& map)
      : BasePlanner(), graph_(graph), map_(map) {}

  std::vector<std::vector<float>> searchPath(std::vector<float> start,
                                             std::vector<float> goal,
                                             float dist_th) const override;
  /**
   * Checks if a line segment is visible in the grid map.
   *
   * This function calls through to the isVisible method of the
   * underlying GridMap object.
   *
   * @param x1 World x coordinate of the start point.
   * @param y1 World y coordinate of the start point.
   * @param x2 World x coordinate of the end point.
   * @param y2 World y coordinate of the end point.
   * @return True if the line segment is visible, false otherwise.
   */
  bool isVisible(float x1, float y1, float x2, float y2) const override {
    return map_.isVisible(x1, y1, x2, y2, 0.01f, true);
  }

 private:
  BaseGraph const& graph_;
  GridMap const& map_;

  std::vector<std::vector<float>> reconstructPath_(
      AStarNode::Ptr current) const;

  bool findNodeOrCreate(BaseGraph& graph, std::vector<float> node,
                        BaseGraphNode::Ptr& result) const ;
};

#endif