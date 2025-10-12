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
      return -1;
    }
  };

  float computeTraversabilityCost(GridMap const& map,
                                  std::vector<std::vector<float>> path,
                                  AgentType agent_type, float lambda_good_trav, float lambda_bad_trav) const {
    
    if(agent_type == AgentType::AERIAL) {return 0.0;} // Aerial is not affected by trav.

    // Check traversability along the whole path
    float total_accident_rate = 0.0;
    float dist_interv = map.resolution;

    for (int i = 1; i < path.size(); i++) {
      float dist_along_path = 0.0;
      float dx = path[i][0] - path[i - 1][0];
      float dy = path[i][1] - path[i - 1][1];
      float total_dist = sqrt(dx * dx + dy * dy);
      while (dist_along_path < total_dist) {
        float x = path[i - 1][0] + dist_along_path * dx / total_dist;
        float y = path[i - 1][1] + dist_along_path * dy / total_dist;

        // Get traversability and add the corresponding accident rate
        if (map.getTraversability(x, y, agent_type) == 0) {
          total_accident_rate += lambda_bad_trav * dist_interv;
        } else {
          total_accident_rate += lambda_good_trav * dist_interv;
        }
        dist_along_path += dist_interv;
      }
    }

    // Calculate the survival function S(pi) = exp(-total_accident_rate)
    float survival_prob = exp(-total_accident_rate);

    // Return the heterogeneous cost: 1 - S(pi)
    return 1 - survival_prob;
  }

  float computeCollisionRate(float dist, float gamma, float d_05) const {
    // This function implements the logistic function for the accident rate
    return 1.0f / (1.0f + exp(gamma * (dist - d_05)));
  }

  float computeCollisionCost(GridMap const& map,
                          std::vector<std::vector<float>> path,
                          AgentType agent_type, float gamma, float d_05) const {

    if (agent_type != AgentType::AERIAL) {return 0.0f;} // Ground is not affected by collision

    // Total accident rate integrated over the path
    float total_accident_rate = 0.0f;
    float dist_interv = map.resolution;

    for (int i = 1; i < path.size(); i++) {
      float dist_along_path = 0.0;
      float dx = path[i][0] - path[i - 1][0];
      float dy = path[i][1] - path[i - 1][1];
      float total_dist = sqrt(dx * dx + dy * dy);
      while (dist_along_path < total_dist) {
        float x = path[i - 1][0] + dist_along_path * dx / total_dist;
        float y = path[i - 1][1] + dist_along_path * dy / total_dist;

        // Get the distance to the nearest obstacle from the Euclidean Distance
        // Field (EDF)
        float dist_to_obstacle = map.getDistance(x, y);

        // Calculate the accident rate using the logistic function
        float lambda_safe = computeCollisionRate(dist_to_obstacle, gamma, d_05);

        // Add the rate times the distance interval to the total
        total_accident_rate += lambda_safe * dist_interv;

        dist_along_path += dist_interv;
      }
    }

    // Calculate the survival probability S(pi) = exp(-total_accident_rate)
    float survival_prob = exp(-total_accident_rate);

    // Return the safety cost: 1 - S(pi)
    return 1 - survival_prob;
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