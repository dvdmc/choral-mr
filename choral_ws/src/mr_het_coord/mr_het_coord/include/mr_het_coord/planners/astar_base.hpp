#ifndef MR_HET_COORD_ASTAR_BASE_HPP
#define MR_HET_COORD_ASTAR_BASE_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

struct AStarNode {
  using Ptr = std::shared_ptr<AStarNode>;

  float x, y;
  float f, g, h;
  AStarNode::Ptr parent;

  AStarNode(float x, float y, float g = 0, float h = 0,
            AStarNode::Ptr parent = nullptr)
      : x(x), y(y), g(g), h(h), f(g + h), parent(parent) {}

  // Comparison operators
  bool operator==(AStarNode const& other) const {
    return std::abs(x - other.x) < 0.0001 && std::abs(y - other.y) < 0.0001;
  }
};

struct CompareNode {
  inline bool operator()(AStarNode::Ptr n1, AStarNode::Ptr n2) { return n1->f > n2->f; }
};

struct FloatPairHash {
  std::size_t operator()(std::pair<float, float> const& p) const {
    // Create a combined hash of two floats using a hash combination formula
    auto hash1 = std::hash<float>{}(p.first);
    auto hash2 = std::hash<float>{}(p.second);
    return hash1 ^ (hash2 << 1);
  }
};

// Heuristic function: Manhattan Distance
double heuristic(std::vector<float> const& p1, std::vector<float> const& p2);

#endif