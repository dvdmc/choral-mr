#include "mr_het_coord/planners/astar_base.hpp"

double heuristic(std::vector<float> const& p1, std::vector<float> const& p2) {
  float dx = p1[0] - p2[0];
  float dy = p1[1] - p2[1];
  return std::sqrt(dx * dx + dy * dy);
}