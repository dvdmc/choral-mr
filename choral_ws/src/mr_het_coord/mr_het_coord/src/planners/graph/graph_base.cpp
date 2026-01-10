#include "mr_het_coord/planners/graph/graph_base.hpp"

BaseGraph::BaseGraph() {}

BaseGraph::~BaseGraph() {}

static float EPSILON = 1e-6f;

void BaseGraph::addNode(BaseGraphNode::Ptr node)
{
  nodes_.push_back(node);
  int newSize = nodes_.size();
  adjacency_.resize(newSize);
  for (auto &row : adjacency_)
  {
    row.resize(newSize, 0);
  }
}

void BaseGraph::addEdge(BaseGraphNode::Ptr from, BaseGraphNode::Ptr to)
{
  if (from->id >= adjacency_.size() || to->id >= adjacency_.size())
  {
    throw std::out_of_range("Node ID out of bounds in adjacency matrix.");
  }
  adjacency_[from->id][to->id] = 1;
  adjacency_[to->id][from->id] = 1;
  from->neighbors.insert(to);
  to->neighbors.insert(from);
}

void BaseGraph::deleteEdge(BaseGraphNode::Ptr from, BaseGraphNode::Ptr to)
{
  if (from->id >= adjacency_.size() || to->id >= adjacency_.size())
  {
    throw std::out_of_range("Node ID out of bounds in adjacency matrix.");
  }
  if (adjacency_[from->id][to->id] == 0)
    return;

  adjacency_[from->id][to->id] = 0;
  adjacency_[to->id][from->id] = 0;
  from->neighbors.erase(to);
  to->neighbors.erase(from);
}

void BaseGraph::clearEdges(BaseGraphNode::Ptr node)
{
  if (node->id >= adjacency_.size())
  {
    throw std::out_of_range("Node ID out of bounds in adjacency matrix.");
  }

  for (int i = 0; i < adjacency_.size(); i++)
  {
    adjacency_[node->id][i] = 0;
    adjacency_[i][node->id] = 0;
  }

  std::vector<BaseGraphNode::Ptr> neighbors_copy(node->neighbors.begin(), node->neighbors.end());
  for (auto neighbor : neighbors_copy)
  {
    neighbor->neighbors.erase(node);
  }

  node->neighbors.clear();
}

std::vector<BaseGraphNode::Ptr> BaseGraph::getNeighborsDist(
    std::vector<float> const &position, float search_radius, bool check_self) const
{
  std::vector<BaseGraphNode::Ptr> neighbors;

  for (auto const &node : nodes_)
  {
    float dist = euclideanDist({node->x, node->y}, position);
    if (dist < search_radius)
    {
      if (check_self && dist < EPSILON)
      {
        continue; // Skip the node itself
      }
      neighbors.push_back(node);
    }
  }
  return neighbors;
}

std::vector<BaseGraphNode::Ptr> BaseGraph::getKNeighbors(
    std::vector<float> const &position, int k, bool check_self) const
{
  std::vector<BaseGraphNode::Ptr> neighbors;

  if (nodes_.empty() || k <= 0)
  {
    return {};
  }

  std::vector<std::pair<float, BaseGraphNode::Ptr>> distances;

  for (auto node : nodes_)
  {
    float dist = euclideanDist({node->x, node->y}, position);
    if (check_self && dist < EPSILON)
    {
      continue; // Skip the node itself
    }
    distances.emplace_back(dist, node);
  }
  if (distances.size() <= k)
  {
    // If we have fewer than k nodes, return all (after sorting)
    std::sort(distances.begin(), distances.end());
  }
  else
  {
    std::nth_element(distances.begin(), distances.begin() + k, distances.end());
    distances.resize(k);
    std::sort(distances.begin(), distances.end()); // Ensure the result is sorted
  }

  // Extract sorted neighbors
  std::vector<BaseGraphNode::Ptr> k_neighbors;
  k_neighbors.reserve(distances.size());
  for (const auto &[dist, node] : distances)
  {
    k_neighbors.push_back(node);
  }

  return k_neighbors;
}

std::vector<BaseGraphNode::Ptr> BaseGraph::getKNeighbors(
  BaseGraphNode::Ptr node, int k, bool check_self, bool only_unconnected) const {

if (nodes_.empty() || k <= 0) {
  return {};
}

std::vector<std::pair<float, BaseGraphNode::Ptr>> distances;
distances.reserve(nodes_.size());

// Fast lookup for existing neighbors (only used if only_unconnected is true)
auto& existing_neighbors = node->neighbors;
for (auto other_node : nodes_) {
  if (check_self && other_node == node) {
    continue;  // Skip self
  }
  
  if (only_unconnected && existing_neighbors.find(other_node) != existing_neighbors.end()) {
    continue;  // Skip already connected nodes
  }

  float dist = euclideanDist({node->x, node->y}, {other_node->x, other_node->y});
  distances.emplace_back(dist, other_node);
}

if (distances.size() <= k) {
  std::sort(distances.begin(), distances.end());
} else {
  std::nth_element(distances.begin(), distances.begin() + k, distances.end());
  distances.resize(k);
  std::sort(distances.begin(), distances.end());  // Ensure final sorting
}

std::vector<BaseGraphNode::Ptr> neighbors;
neighbors.reserve(distances.size());
for (const auto& [dist, other_node] : distances) {
  neighbors.push_back(other_node);
}

return neighbors;
}

std::vector<std::pair<BaseGraphNode::Ptr, BaseGraphNode::Ptr>>
BaseGraph::getKPotentialConnections(std::vector<BaseGraphNode::Ptr> nodes1,
                                    std::vector<BaseGraphNode::Ptr> nodes2,
                                    int k) const
{
  if (nodes1.empty() || nodes2.empty() || k <= 0)
  {
    return {};
  }

  std::vector<std::pair<std::pair<int, int>, float>> distances;
  distances.reserve(nodes1.size() * nodes2.size());

  // Compute pairwise distances
  for (int i = 0; i < nodes1.size(); i++)
  {
    for (int j = 0; j < nodes2.size(); j++)
    {
      float dist = euclideanDist({nodes1[i]->x, nodes1[i]->y},
                                  {nodes2[j]->x, nodes2[j]->y});
      distances.emplace_back(std::make_pair(i, j), dist);
    }
  }

  if (distances.size() > k)
  {
    std::nth_element(distances.begin(), distances.begin() + k, distances.end(),
                     [](const auto &a, const auto &b)
                     {
                       return a.second < b.second;
                     });
    distances.resize(k); // Keep only the top k elements
  }
  else
  {
    std::sort(distances.begin(), distances.end(),
              [](const auto &a, const auto &b)
              {
                return a.second < b.second;
              });
  }

  // Collect k closest pairs
  std::vector<std::pair<BaseGraphNode::Ptr, BaseGraphNode::Ptr>> k_pairs;
  k_pairs.reserve(k);
  for (int i = 0; i < std::min(k, (int)distances.size()); i++)
  {
    k_pairs.emplace_back(nodes1[distances[i].first.first],
                         nodes2[distances[i].first.second]);
  }

  return k_pairs;
}

BaseGraph BaseGraph::copy() const
{
  BaseGraph new_graph;
  std::unordered_map<BaseGraphNode::Ptr, BaseGraphNode::Ptr> node_map;

  for (auto const &node : nodes_)
  {
    BaseGraphNode::Ptr new_node =
        std::make_shared<BaseGraphNode>(node->id, node->x, node->y);
    new_graph.addNode(new_node);
    node_map[node] = new_node; // Map original node to new node
  }

  for (const auto &node : nodes_)
  {
    BaseGraphNode::Ptr new_node = node_map[node];

    for (const auto &neighbor : node->neighbors)
    {
      if (node->id < neighbor->id)
      { // Prevent duplicate edges
        auto it = node_map.find(neighbor);
        if (it != node_map.end())
        {
          new_graph.addEdge(new_node, it->second);
        }
      }
    }
  }

  return new_graph;
}

void BaseGraph::dfs(int node, std::vector<bool> &visited,
                    std::vector<BaseGraphNode::Ptr> &component) const
{
  std::stack<int> stack;
  stack.push(node);

  while (!stack.empty())
  {
    int current = stack.top();
    stack.pop();

    if (!visited[current])
    {
      visited[current] = true;
      component.push_back(nodes_[current]);

      for (auto neighbor : nodes_[current]->neighbors)
      {
        if (!visited[neighbor->id])
        {
          stack.push(neighbor->id);
        }
      }
    }
  }
}

std::vector<ConnectedComponent> BaseGraph::findConnectedComponents() const
{
  int n = nodes_.size();
  std::vector<bool> visited(n, false);
  std::vector<ConnectedComponent> components;
  int id = 0;
  for (int i = 0; i < n; ++i)
  {
    if (!visited[i])
    {
      std::vector<BaseGraphNode::Ptr> component;
      dfs(i, visited, component); // Find one connected component starting from i
      components.push_back(ConnectedComponent(id++, component));
    }
  }

  return components;
}