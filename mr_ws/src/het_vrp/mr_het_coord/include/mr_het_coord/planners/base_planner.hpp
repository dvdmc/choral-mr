#ifndef MR_HET_COORD_BASE_PLANNER_HPP
#define MR_HET_COORD_BASE_PLANNER_HPP

#include <vector>

#include "mr_het_coord/map/map.hpp"

class BasePlanner {
 public:
  virtual std::vector<std::vector<float>> searchPath(std::vector<float> start,
                                                     std::vector<float> goal,
                                                     float dist_th) const = 0;

  virtual bool isVisible(float x1, float y1, float x2, float y2) const = 0;

  std::vector<std::vector<float>> invertPath(
      const std::vector<std::vector<float>>& path) const {
    std::vector<std::vector<float>> inverted_path;
    for (int i = path.size() - 1; i >= 0; --i) {
      inverted_path.push_back(path[i]);
    }
    return inverted_path;
  }

  float computeDistance(std::vector<std::vector<float>> path) const {
    // If path found
    if (!path.empty()) {
      // Compute distance
      float distance = 0.0;
      for (int i = 1; i < path.size(); ++i) {
        float x = path[i][0] - path[i - 1][0];
        float y = path[i][1] - path[i - 1][1];
        distance += sqrt(x * x + y * y);
      }
      return distance;
    } else {
      // Path not found. Return -1
      return -1;
    }
  };

  float computeTraversabilityCost(const GridMap& map,
                                  const std::vector<std::vector<float>>& path,
                                  AgentType agent_type, float lambda_good_trav,
                                  float lambda_bad_trav) const {
    if (path.empty()) return -1.0f;
    if (agent_type == AgentType::AERIAL)
    {
      return 0.0f;  // Aerial unaffected by traversability
    }

    float total_accident_rate = 0.0f;
    const float dist_interv = map.resolution;

    for (size_t i = 1; i < path.size(); ++i) {
      float dx = path[i][0] - path[i - 1][0];
      float dy = path[i][1] - path[i - 1][1];
      float total_dist = std::sqrt(dx * dx + dy * dy);

      for (float s = 0.0f; s < total_dist; s += dist_interv) {
        float x = path[i - 1][0] + s * dx / total_dist;
        float y = path[i - 1][1] + s * dy / total_dist;

        float lambda = (map.getTraversability(x, y, agent_type) == 0)
                           ? lambda_bad_trav
                           : lambda_good_trav;

        total_accident_rate += lambda * dist_interv;
      }
    }

    float survival_prob = std::exp(-total_accident_rate);
    return 1.0f - survival_prob;
  }

  float computeCollisionRate(float dist, float gamma, float d_05) const {
    // Logistic function for collision rate
    return 1.0f / (1.0f + exp(gamma * (dist - d_05)));
  }

  float computeCollisionCost(const GridMap& map,
                             const std::vector<std::vector<float>>& path,
                             AgentType agent_type, float gamma,
                             float d_05) const {
    if (path.empty()) return -1.0f;
    if (agent_type != AgentType::AERIAL)
    {
      return 0.0f;  // Ground unaffected by collision
    }

    float total_accident_rate = 0.0f;
    const float dist_interv = map.resolution;

    for (size_t i = 1; i < path.size(); ++i) {
      float dx = path[i][0] - path[i - 1][0];
      float dy = path[i][1] - path[i - 1][1];
      float total_dist = std::sqrt(dx * dx + dy * dy);

      for (float s = 0.0f; s < total_dist; s += dist_interv) {
        float x = path[i - 1][0] + s * dx / total_dist;
        float y = path[i - 1][1] + s * dy / total_dist;

        float dist_to_obstacle = map.getDistance(x, y);
        float lambda = computeCollisionRate(dist_to_obstacle, gamma, d_05);
        total_accident_rate += lambda * dist_interv;
      }
    }

    float survival_prob = std::exp(-total_accident_rate);
    return 1.0f - survival_prob;
  }

  float computeSafetyCost(const GridMap& map,
                          const std::vector<std::vector<float>>& path,
                          AgentType agent_type, float lambda_good_trav,
                          float lambda_bad_trav, float gamma,
                          float d_05) const {
    // Return 0 if no path or trivial path
    if (path.empty()) return -1.0f;

    float total_accident_rate = 0.0f;
    float dist_interv = map.resolution;

    for (int i = 1; i < path.size(); ++i) {
      float dx = path[i][0] - path[i - 1][0];
      float dy = path[i][1] - path[i - 1][1];
      float total_dist = sqrt(dx * dx + dy * dy);
      float dist_along_path = 0.0f;

      while (dist_along_path < total_dist) {
        float x = path[i - 1][0] + dist_along_path * dx / total_dist;
        float y = path[i - 1][1] + dist_along_path * dy / total_dist;

        float local_rate = 0.0f;

        // Traversability-related accident rate (for ground agents)
        if (agent_type != AgentType::AERIAL) {
          if (map.getTraversability(x, y, agent_type) == 0) {
            local_rate += lambda_bad_trav;
          } else {
            local_rate += lambda_good_trav;
          }
        }

        // Collision-related accident rate (for aerial agents)
        if (agent_type == AgentType::AERIAL) {
          float dist_to_obstacle = map.getDistance(x, y);
          local_rate += computeCollisionRate(dist_to_obstacle, gamma, d_05);
        }

        total_accident_rate += local_rate * dist_interv;
        dist_along_path += dist_interv;
      }
    }

    // Poisson survival probability: S(pi) = exp(-sum(local_rate * dist_interv))
    float survival_prob = exp(-total_accident_rate);

    // Safety cost: 1 - S(pi)
    return 1.0f - survival_prob;
  }

  std::vector<std::vector<float>> simplifyPath(
      std::vector<std::vector<float>> const& path) const {
    if (path.size() <= 2) {
      return path;  // No simplification needed if there are only two points
    }

    std::vector<std::vector<float>> simplifiedPath;
    simplifiedPath.push_back(path[0]);  // Always keep the start point

    int lastKeptPoint = 0;  // Index of the last kept point

    for (int i = 1; i < path.size(); i++) {
      // Check if the next point is visible from the last kept point
      // TODO: Below is not good practice since we are modifying i inside the
      // for
      if (!isVisible(path[lastKeptPoint][0], path[lastKeptPoint][1], path[i][0],
                     path[i][1])) {
        simplifiedPath.push_back(
            path[i - 1]);       // Add the last point if not visible
        lastKeptPoint = i - 1;  // Update the last kept point
      }
    }

    // Always add the last point
    simplifiedPath.push_back(path[path.size() - 1]);

    return simplifiedPath;
  }

  float euclideanDistance(std::vector<float> const& p1,
                          std::vector<float> const& p2) const {
    float dx = p1[0] - p2[0];
    float dy = p1[1] - p2[1];
    return std::sqrt(dx * dx + dy * dy);
  }
};

#endif