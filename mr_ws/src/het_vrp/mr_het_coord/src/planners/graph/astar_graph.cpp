#include "mr_het_coord/planners/graph/astar_graph.hpp"

#include <unordered_set>

std::vector<std::vector<float>> AStarGraph::reconstructPath_(
    AStarNode::Ptr current) const {
  std::vector<std::vector<float>> path;
  while (current != nullptr) {
    path.push_back({current->x, current->y});
    current = current->parent;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

bool AStarGraph::findNodeOrCreate(BaseGraph& graph, std::vector<float> node,
                                  BaseGraphNode::Ptr& result) const {
  std::vector<BaseGraphNode::Ptr> neighbors =
      graph.getKNeighbors(node, 5, false);

  std::vector<BaseGraphNode::Ptr> to_connect;
  for (auto neighbor : neighbors) {
    // If the position is the same as a node in the graph,
    // directly choose it
    if (std::abs(neighbor->x - node[0]) < 1e-6 &&
        std::abs(neighbor->y - node[1]) < 1e-6) {
      result = neighbor;
      return true;
    }

    // Otherwise, check the visibility and add it to the to_connect
    if (isVisible(node[0], node[1], neighbor->x, neighbor->y)) {
      to_connect.push_back(neighbor);
    }
  }

  if (to_connect.size() == 0) {
    return false;
  }

  BaseGraphNode::Ptr new_node =
      std::make_shared<BaseGraphNode>(graph.size(), node[0], node[1]);
  graph.addNode(new_node);
  for (auto neighbor : to_connect) {
    graph.addEdge(new_node, neighbor);
  }
  result = new_node;
  return true;
}

std::vector<std::vector<float>> AStarGraph::searchPath(std::vector<float> start,
                                                       std::vector<float> goal,
                                                       float dist_th) const {
  if (euclideanDistance(start, goal) < dist_th ||
      isVisible(start[0], start[1], goal[0], goal[1])) {
    return {start, goal};  // Direct path
  }

  BaseGraph search_graph = graph_;  // Assume graph is mutable or lightweight

  std::priority_queue<AStarGraphNode::Ptr, std::vector<AStarGraphNode::Ptr>,
                      CompareNode>
      open_set;
  std::unordered_set<int> closed_set;  // Store node IDs instead of coordinates
  std::unordered_map<int, double>
      g_score_map;  // Track best g values for nodes in open set

  BaseGraphNode::Ptr goal_node, start_node;
  if (!findNodeOrCreate(search_graph, goal, goal_node) ||
      !findNodeOrCreate(search_graph, start, start_node)) {
    std::cout << "Start or goal node not connected to graph" << std::endl;
    return {};
  }

  open_set.push(std::make_shared<AStarGraphNode>(
      start[0], start[1], 0, heuristic(start, goal), nullptr, start_node));
  g_score_map[start_node->id] = 0;

  while (!open_set.empty()) {
    AStarGraphNode::Ptr current = open_set.top();
    open_set.pop();

    if (euclideanDistance({current->x, current->y}, goal) < dist_th) {
      auto path = reconstructPath_(current);
      if (path.back() != goal) path.push_back(goal);  // Ensure goal is included
      return simplifyPath(path);
    }

    closed_set.insert(current->graph_node->id);

    for (auto& neighbor : current->graph_node->neighbors) {
      if (closed_set.find(neighbor->id) != closed_set.end()) {
        continue;
      }
      double tentative_g = current->g + std::hypot(neighbor->x - current->x,
                                                   neighbor->y - current->y);

      if (g_score_map.find(neighbor->id) == g_score_map.end() ||
          tentative_g < g_score_map[neighbor->id]) {
        g_score_map[neighbor->id] = tentative_g;
        open_set.push(std::make_shared<AStarGraphNode>(
            neighbor->x, neighbor->y, tentative_g,
            heuristic({neighbor->x, neighbor->y}, goal), current, neighbor));
      }
    }
  }

  std::cout << "No path found" << std::endl;
  return {};
}