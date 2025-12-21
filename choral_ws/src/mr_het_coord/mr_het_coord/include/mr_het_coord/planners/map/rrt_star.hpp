#ifndef MR_HET_COORD_RRT_STAR_HPP
#define MR_HET_COORD_RRT_STAR_HPP

#include "mr_het_coord/map/map.hpp"
#include "mr_het_coord/planners/base_planner.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

struct RRTNode {
  using Ptr = std::shared_ptr<RRTNode>;

  float x, y;
  float g;
  Ptr parent;

  RRTNode(float x, float y, float g, Ptr parent = nullptr)
      : x(x), y(y), g(g), parent(parent) {}

  bool operator==(RRTNode const& other) const {
    return std::abs(x - other.x) < 0.0001 && std::abs(y - other.y) < 0.0001;
  }
};

class RRTStar : public BasePlanner {
 public:
  RRTStar(GridMap const& map, float step_size, int max_samples,
          float search_radius, std::mt19937& rng, bool anytime = false)
      : BasePlanner(),
        map_(map),
        step_size_(step_size),
        visibility_step_(step_size / 10.0f),
        max_samples_(max_samples),
        search_radius_(search_radius),
        anytime_(anytime),
        minXY_(map.getMinXY()),
        maxXY_(map.getMaxXY()),
        rng_(rng) {}

  std::vector<std::vector<float>> searchPath(std::vector<float> start,
                                             std::vector<float> goal,
                                             float dist_th) const override;

  std::vector<std::vector<float>> searchPathMultiGoal(
      std::vector<float> start, std::vector<std::vector<float>> goals,
      float dist_th) const;

  bool isVisible(float x1, float y1, float x2, float y2) const override {
    return map_.isVisible(x1, y1, x2, y2, visibility_step_, true);
  }

 private:
  GridMap const& map_;
  float step_size_;
  float visibility_step_;
  int max_samples_;
  float search_radius_;
  bool anytime_;
  std::vector<float> minXY_;
  std::vector<float> maxXY_;
  std::mt19937 &rng_;

  /**
   * Reconstruct the path from the given node to the root of the tree.
   *
   * Given a node in the tree, this function will trace back the path to the
   * root node and return it as a sequence of points in the map.
   *
   * @param goal_node The node from which to reconstruct the path.
   * @return The reconstructed path, as a sequence of points in the map.
   */
  std::vector<std::vector<float>> reconstructPath_(
      RRTNode::Ptr goal_node) const;

  /**
   * @brief Generates a random sample in the map.
   *
   * This function generates a random point (x, y) in the map, ensuring that the
   * point is valid (i.e., not an obstacle). The point is generated using
   * uniform distributions for x and y within the map's bounding box.
   *
   * @param rng The random number generator used to sample the point.
   * @return A 2D vector containing the x and y coordinates of the sampled
   * point.
   */
  std::vector<float> randomSample_(std::mt19937& rng) const;

  /**
   * Samples a random point within a bounding box defined by the start and goal
   * positions with a given variance scale. The sampled point is guaranteed to
   * be valid on the map.
   *
   * @param start The start position.
   * @param goal The goal position.
   * @param variance_scale A scale factor for the variance of the bounding box.
   * @param rng A random number generator.
   * @return The sampled point as a 2D vector.
   */
  std::vector<float> informedRandomSample_(std::vector<float> start,
                                           std::vector<float> goal,
                                           float variance_scale,
                                           std::mt19937& rng) const;

  /**
   * @brief Finds the nearest node in the tree to the given position.
   *
   * This function iterates over all nodes in the provided tree and calculates
   * their Euclidean distance to the specified position. It returns the node
   * that is closest to the position.
   *
   * @param tree A vector of pointers to RRTNodes representing the tree.
   * @param position A vector of floats representing the (x, y) coordinates
   *                 of the position to compare against.
   * @return A pointer to the RRTNode that is nearest to the given position.
   */
  RRTNode::Ptr getNearestNode_(std::vector<RRTNode::Ptr> const& tree,
                               std::vector<float> const& position) const;

  /**
   * @brief Get all nodes in the tree that are within search_radius of the given
   * position.
   *
   * @param tree The tree to search
   * @param position The position to search around
   * @return A vector of all nodes in the tree that are within search_radius of the
   * given position.
   */               
  std::vector<RRTNode::Ptr> nearNeighbors_(
      std::vector<RRTNode::Ptr> const& tree,
      std::vector<float> const& position) const;

  void clearTree_(std::vector<RRTNode::Ptr>& tree) const;
};

#endif