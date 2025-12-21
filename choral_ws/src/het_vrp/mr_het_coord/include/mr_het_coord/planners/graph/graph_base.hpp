#ifndef MR_HET_COORD_GRAPH_BASE_HPP
#define MR_HET_COORD_GRAPH_BASE_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <vector>

struct BaseGraphNode {
  using Ptr = std::shared_ptr<BaseGraphNode>;

  BaseGraphNode(int id, float x, float y) : id(id), x(x), y(y) {}

  int id;
  float x, y;

  // Custom hash function for BaseGraphNode pointers
  struct BaseGraphNodeHash {
    std::size_t operator()(Ptr const& node) const {
      return std::hash<BaseGraphNode*>()(node.get());
    }
  };

  // Custom equality function for BaseGraphNode pointers
  struct BaseGraphNodeEqual {
    bool operator()(Ptr const& node1, Ptr const& node2) const {
      return node1->id == node2->id;
    }
  };

  // Unordered set of neighbor pointers using custom hash and equality functions
  std::unordered_set<Ptr, BaseGraphNodeHash, BaseGraphNodeEqual> neighbors;
};

struct ConnectedComponent {
  int id;
  std::vector<BaseGraphNode::Ptr> nodes;

  ConnectedComponent(int id, std::vector<BaseGraphNode::Ptr> nodes) : id(id), nodes(nodes) {}
};

class BaseGraph {
  std::vector<BaseGraphNode::Ptr> nodes_;
  std::vector<std::vector<int>> adjacency_;

 public:
  BaseGraph();

  ~BaseGraph();


  /**
   * Debug function to print adjacency matrix and neighbor information for two nodes
   *
   * @param from Node index
   * @param to Node index
   */
  void isConnected(int from, int to) const {
    std::cout << "Adj: " << adjacency_[from][to] << std::endl;
    std::cout << "Adj symm: " << adjacency_[to][from] << std::endl;
    std::cout << "Neigh from: ";
    for (auto neigh : nodes_[from]->neighbors) {
      std::cout << neigh->id << " ";
    }
    std::cout << std::endl;
    std::cout << "Neigh to: ";
    for (auto neigh : nodes_[to]->neighbors) {
      std::cout << neigh->id << " ";
    }
    std::cout << std::endl;
  }

  /**
   * @brief Clears the graph by removing all nodes and edges.
   *
   * This function empties the nodes and adjacency matrix, effectively
   * resetting the graph to an initial state with no nodes or connections.
   */
  void clearGraph() {
    nodes_.clear();
    adjacency_.clear();
  }

  int size() const { return nodes_.size(); }

  std::vector<BaseGraphNode::Ptr> getNodes() const { return nodes_; }

  /**
   * @brief Add a node to the graph
   *
   * @param node The node to add
   *
   * This will create a new row and column in the adjacency matrix and set all
   * values to 0.
   */
  void addNode(BaseGraphNode::Ptr node);

  /**
   * @brief Adds an edge between two nodes in the graph.
   *
   * This function sets the corresponding value in the adjacency matrix to 1,
   * indicating that there is an edge between the two nodes. It also adds each node
   * to the other's neighbor set.
   *
   * @param from The node that the edge starts from.
   * @param to The node that the edge ends at.
   */
  void addEdge(BaseGraphNode::Ptr from, BaseGraphNode::Ptr to);

  /**
   * Deletes an edge between two nodes in the graph
   *
   * @param from The node that the edge is going away from
   * @param to The node that the edge is going towards
   *
   * Deletes the edge by setting the corresponding adjacency matrix
   * elements to 0 and removing the edge from the neighbors of both
   * nodes.
   */
  void deleteEdge(BaseGraphNode::Ptr from, BaseGraphNode::Ptr to);

  /**
   * @brief Clears all edges connected to a given node in the graph.
   *
   * This function removes all edges associated with the specified node by setting
   * the corresponding entries in the adjacency matrix to 0. It also removes the
   * node from the neighbor set of all its connected nodes and clears the node's
   * own neighbor set.
   *
   * @param node The node whose edges are to be cleared.
   */
  void clearEdges(BaseGraphNode::Ptr node);

  /**
   * @brief Returns all nodes within a given search radius from a given position.
   *
   * This function iterates over all nodes in the graph and checks if the
   * Euclidean distance between the node's position and the specified position is
   * less than the search radius. If so, it adds the node to the result vector.
   * If `check_self` is true, it skips the node itself if the distance is
   * approximately 0.
   *
   * @param position The position to search around.
   * @param search_radius The search radius.
   * @param check_self Whether to skip the node itself if the distance is 0.
   * @return A vector of all nodes within the search radius.
   */
  std::vector<BaseGraphNode::Ptr> getNeighborsDist(
      std::vector<float> const& position, float search_radius, bool check_self = true) const;

  /**
   * @brief Finds the k nearest nodes to a given position.
   *
   * This function uses a priority queue to keep track of the k nearest nodes to
   * the given position. It iterates over all nodes in the graph and adds them to
   * the priority queue if there are less than k nodes in the queue. If the queue
   * is full, it checks if the current node is closer to the position than the
   * node with the highest distance in the queue. If so, it replaces that node.
   * Finally, it pops all nodes from the queue and returns them in a vector.
   * @param position The position to search around.
   * @param k The number of nearest neighbors to return.
   * @param check_self Whether to skip the node itself if it is among the k
   * nearest neighbors.
   * @param only_unconnected Whether to only return neighbors that are not
   * @return A vector of the k nearest neighbors.
   */
  std::vector<BaseGraphNode::Ptr> getKNeighbors(
    std::vector<float> const& position, int k, bool check_self) const;

  std::vector<BaseGraphNode::Ptr> getKNeighbors(
    BaseGraphNode::Ptr node, int k, bool check_self, bool only_unconnected=false) const;

  /**
   * @brief Find the k closest pairs of nodes between two sets of nodes in the graph.
   *
   * This function computes the Euclidean distance between all pairs of nodes
   * between two sets of nodes and returns the k closest pairs.  The pairs are
   * sorted by distance and only the k closest pairs are returned.
   *
   * @param nodes1 The first set of nodes
   * @param nodes2 The second set of nodes
   * @param k The number of closest pairs to return
   *
   * @return A vector of pairs of BaseGraphNode pointers, representing the k
   * closest pairs of nodes between the two sets.  The pairs are sorted by
   * distance.
   */
  std::vector<std::pair<BaseGraphNode::Ptr, BaseGraphNode::Ptr>>
  getKPotentialConnections(std::vector<BaseGraphNode::Ptr> nodes1,
                           std::vector<BaseGraphNode::Ptr> nodes2, int k) const;

  /**
   * Creates a deep copy of the current graph.
   * 
   * This function generates a new instance of BaseGraph, duplicating all the nodes
   * and edges from the original graph. It maintains the same structure and connectivity
   * as the original graph. The nodes are duplicated by creating new instances, and
   * the edges are re-established using a mapping from the original nodes to the new nodes.
   * 
   * @return A new BaseGraph object that is a deep copy of the current graph.
   */
  BaseGraph copy() const;

  /**
   * @brief Performs a depth-first search on the graph, starting from the given node
   *
   * @param node The index of the node to start the search from
   * @param visited A boolean vector of size nodes_.size() that keeps track of
   * which nodes have been visited
   * @param component A vector to store the nodes of the connected component
   *
   * This function is a helper function for findConnectedComponents_.
   */
  void dfs(int node, std::vector<bool>& visited,
           std::vector<BaseGraphNode::Ptr>& component) const;

  /**
   * @brief Finds all connected components in the graph
   *
   * @return A vector of vectors of node pointers, where each sub-vector is a
   * connected component
   *
   * This function performs a depth-first search on the graph to find all of the
   * connected components. The result is a vector of vectors, where each sub-vector
   * is a connected component. The order of the sub-vectors is undefined.
   */
  std::vector<ConnectedComponent> findConnectedComponents() const;

  float euclideanDist(std::vector<float> const& p1,
                      std::vector<float> const& p2) const {
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    return std::sqrt(dx * dx + dy * dy);
  }

  void printGraph() const {
    for (int i = 0; i < nodes_.size(); i++) {
      std::cout << "Node " << i << " connected to: ";
      for (auto neightbor : nodes_[i]->neighbors) {
        std::cout << neightbor->id << " ";
      }
      std::cout << std::endl;
    }
  }
 private:
};

#endif