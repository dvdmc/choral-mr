#include "mr_het_coord/planners/map/rrt_star.hpp"

std::vector<std::vector<float>> RRTStar::reconstructPath_(
    RRTNode::Ptr goal_node) const {
  std::vector<std::vector<float>> path;
  for (RRTNode::Ptr node = goal_node; node != nullptr; node = node->parent) {
    path.push_back({node->x, node->y});
  }
  std::reverse(path.begin(), path.end());

  return path;
}

std::vector<float> RRTStar::randomSample_(std::mt19937& rng) const {
  std::uniform_real_distribution<float> x_dist(minXY_[0], maxXY_[0]);
  std::uniform_real_distribution<float> y_dist(minXY_[1], maxXY_[1]);
  float x = x_dist(rng);
  float y = y_dist(rng);
  while (!map_.isValid(x, y, true)) {
    x = x_dist(rng);
    y = y_dist(rng);
  }
  return {x, y};
}

std::vector<float> RRTStar::informedRandomSample_(std::vector<float> start,
                                                  std::vector<float> goal,
                                                  float variance_scale,
                                                  std::mt19937& rng) const {
  // Determine the bounding box defined by the start and goal
  float min_x = std::min(start[0], goal[0]) - 1.0f * variance_scale;
  float max_x = std::max(start[0], goal[0]) + 1.0f * variance_scale;
  float min_y = std::min(start[1], goal[1]) - 1.0f * variance_scale;
  float max_y = std::max(start[1], goal[1]) + 1.0f * variance_scale;

  // Saturate with map limits
  min_x = std::max(minXY_[0], min_x);
  max_x = std::min(maxXY_[0], max_x);
  min_y = std::max(minXY_[1], min_y);
  max_y = std::min(maxXY_[1], max_y);

  // Uniform distributions for x and y within the bounding box
  std::uniform_real_distribution<float> x_dist(min_x, max_x);
  std::uniform_real_distribution<float> y_dist(min_y, max_y);

  // Random generator
  std::random_device rd;
  std::mt19937 gen(rd());

  float x = x_dist(gen);
  float y = y_dist(gen);

  // Ensure the sampled point is valid on the map
  int counter = 1;
  while (!map_.isValid(x, y, true)) {
    // Increase bbox if sample is not found
    if (counter % 10 == 0) {
      variance_scale *= 1.2;
      min_x -= 1.0f * variance_scale;
      max_x += 1.0f * variance_scale;
      min_y -= 1.0f * variance_scale;
      max_y += 1.0f * variance_scale;

      // Saturate with map limits
      min_x = std::max(minXY_[0], min_x);
      max_x = std::min(maxXY_[0], max_x);
      min_y = std::max(minXY_[1], min_y);
      max_y = std::min(maxXY_[1], max_y);

      // Uniform distributions for x and y within the bounding box
      x_dist = std::uniform_real_distribution<float>(min_x, max_x);
      y_dist = std::uniform_real_distribution<float>(min_y, max_y);
    }

    x = x_dist(gen);
    y = y_dist(gen);
    counter++;
  }

  return {x, y};
}

RRTNode::Ptr RRTStar::getNearestNode_(
    std::vector<RRTNode::Ptr> const& tree,
    std::vector<float> const& position) const {
  RRTNode::Ptr nearest = tree[0];
  float min_dist = euclideanDistance({nearest->x, nearest->y}, position);

  for (auto node : tree) {
    float dist = euclideanDistance({node->x, node->y}, position);
    if (dist < min_dist) {
      nearest = node;
      min_dist = dist;
    }
  }
  return nearest;
}

std::vector<RRTNode::Ptr> RRTStar::nearNeighbors_(
    std::vector<RRTNode::Ptr> const& tree,
    std::vector<float> const& position) const {
  std::vector<RRTNode::Ptr> points_in_radius;

  for (auto node : tree) {
    if (euclideanDistance({node->x, node->y}, position) < search_radius_) {
      points_in_radius.push_back(node);
    }
  }
  return points_in_radius;
}

std::vector<std::vector<float>> RRTStar::searchPath(std::vector<float> start,
                                                    std::vector<float> goal,
                                                    float dist_th) const {
  std::vector<RRTNode::Ptr> tree;
  std::mt19937 rng(std::random_device{}());

  RRTNode::Ptr start_node =
      std::make_shared<RRTNode>(start[0], start[1], 0, nullptr);
  tree.push_back(start_node);

  RRTNode::Ptr best_goal_node = nullptr;
  float best_cost = std::numeric_limits<float>::max();

  // If goal is directly visible, connect
  if (isVisible(start[0], start[1], goal[0], goal[1])) {
    RRTNode::Ptr goal_node = std::make_shared<RRTNode>(
        goal[0], goal[1],
        start_node->g + euclideanDistance({start_node->x, start_node->y}, goal),
        start_node);
    tree.push_back(goal_node);
    std::vector<std::vector<float>> path = reconstructPath_(goal_node);
    clearTree_(tree);
    return simplifyPath(path);
  }

  for (int i = 0; i < max_samples_; ++i) {
    std::vector<float> random_point =
        informedRandomSample_(start, goal, 0.01 * i, rng);

    // Get nearest node in the tree
    RRTNode::Ptr nearest_node = getNearestNode_(tree, random_point);

    // Step towards the random point
    float theta = atan2(random_point[1] - nearest_node->y,
                        random_point[0] - nearest_node->x);
    std::vector<float> new_point = {nearest_node->x + step_size_ * cos(theta),
                                    nearest_node->y + step_size_ * sin(theta)};

    if (!isVisible(nearest_node->x, nearest_node->y, new_point[0],
                   new_point[1])) {
      continue;
    }

    // Create new node
    float new_cost =
        nearest_node->g +
        euclideanDistance({nearest_node->x, nearest_node->y}, new_point);
    RRTNode::Ptr new_node = std::make_shared<RRTNode>(
        new_point[0], new_point[1], new_cost, nearest_node);

    // Add the new node to the tree
    tree.push_back(new_node);

    // Rewire tree
    std::vector<RRTNode::Ptr> neighbors = nearNeighbors_(tree, new_point);
    for (auto neighbor : neighbors) {
      float potential_cost =
          new_cost + euclideanDistance(new_point, {neighbor->x, neighbor->y});
      if (potential_cost < neighbor->g &&
          isVisible(new_point[0], new_point[1], neighbor->x, neighbor->y)) {
        neighbor->parent = new_node;
        neighbor->g = potential_cost;
      }
    }

    // Check if it's near the goal
    if (euclideanDistance(new_point, goal) < dist_th ||
        isVisible(new_point[0], new_point[1], goal[0], goal[1])) {
      if (new_cost < best_cost) {
        best_cost = new_cost;
        best_goal_node = new_node;
      }
      if (!anytime_) {
        // Add goal node
        RRTNode::Ptr goal_node = std::make_shared<RRTNode>(
            goal[0], goal[1],
            best_goal_node->g +
                euclideanDistance({best_goal_node->x, best_goal_node->y}, goal),
            best_goal_node);
        tree.push_back(goal_node);
        std::vector<std::vector<float>> path = reconstructPath_(goal_node);
        clearTree_(tree);
        return simplifyPath(path);
      }
    }
  }

  // Return the best found path
  if (best_goal_node) {
    // Add goal node
    RRTNode::Ptr goal_node = std::make_shared<RRTNode>(
        goal[0], goal[1],
        best_goal_node->g +
            euclideanDistance({best_goal_node->x, best_goal_node->y}, goal),
        best_goal_node);
    tree.push_back(goal_node);
    std::vector<std::vector<float>> path = reconstructPath_(goal_node);
    clearTree_(tree);
    return simplifyPath(path);
  }

  // std::cout << "No path found after all samples! " << std::endl;
  return {};  // Return an empty path if no valid path found
}

void RRTStar::clearTree_(std::vector<RRTNode::Ptr>& tree) const {
  tree.clear();
}

std::vector<std::vector<float>> RRTStar::searchPathMultiGoal(
    std::vector<float> start, std::vector<std::vector<float>> goals,
    float dist_th) const {
  std::vector<RRTNode::Ptr> tree;
  std::mt19937 rng(std::random_device{}());

  RRTNode::Ptr start_node =
      std::make_shared<RRTNode>(start[0], start[1], 0, nullptr);
  tree.push_back(start_node);

  RRTNode::Ptr best_goal_node = nullptr;
  float best_cost = std::numeric_limits<float>::max();

  // If any goal is directly visible, connect
  for (auto goal : goals) {
    if (isVisible(start[0], start[1], goal[0], goal[1])) {
      RRTNode::Ptr goal_node = std::make_shared<RRTNode>(
          goal[0], goal[1],
          start_node->g + euclideanDistance({start_node->x, start_node->y}, goal),
          start_node);
      tree.push_back(goal_node);
      std::vector<std::vector<float>> path = reconstructPath_(goal_node);
      clearTree_(tree);
      return simplifyPath(path);
    }
  }

  for (int i = 0; i < max_samples_; ++i) {
    std::vector<float> random_point = randomSample_(rng);

    // Get nearest node in the tree
    RRTNode::Ptr nearest_node = getNearestNode_(tree, random_point);

    // Step towards the random point
    float theta = atan2(random_point[1] - nearest_node->y,
                        random_point[0] - nearest_node->x);
    std::vector<float> new_point = {nearest_node->x + step_size_ * cos(theta),
                                    nearest_node->y + step_size_ * sin(theta)};
    // std::vector<float> new_point = random_point;

    if (!isVisible(nearest_node->x, nearest_node->y, new_point[0],
                   new_point[1])) {
      continue;
    }

    // Create new node
    float new_cost =
        nearest_node->g +
        euclideanDistance({nearest_node->x, nearest_node->y}, new_point);
    RRTNode::Ptr new_node = std::make_shared<RRTNode>(
        new_point[0], new_point[1], new_cost, nearest_node);

    // Add the new node to the tree
    tree.push_back(new_node);

    // Rewire tree
    std::vector<RRTNode::Ptr> neighbors = nearNeighbors_(tree, new_point);
    for (auto neighbor : neighbors) {
      float potential_cost =
          new_cost + euclideanDistance(new_point, {neighbor->x, neighbor->y});
      if (potential_cost < neighbor->g &&
          isVisible(new_point[0], new_point[1], neighbor->x, neighbor->y)) {
        neighbor->parent = new_node;
        neighbor->g = potential_cost;
      }
    }

    // Check if it's near any goal
    for (auto goal : goals) {
      if (euclideanDistance(new_point, goal) < dist_th ||
          isVisible(new_point[0], new_point[1], goal[0], goal[1])) {
        if (new_cost < best_cost) {
          best_cost = new_cost;
          best_goal_node = new_node;
        }
        if (!anytime_) {
          // Add goal node
          RRTNode::Ptr goal_node = std::make_shared<RRTNode>(
              goal[0], goal[1],
              best_goal_node->g +
                  euclideanDistance({best_goal_node->x, best_goal_node->y}, goal),
              best_goal_node);
          tree.push_back(goal_node);
          std::vector<std::vector<float>> path = reconstructPath_(goal_node);
          clearTree_(tree);
          return simplifyPath(path);
        }
      }
    }
  }

  // Return the best found path
  if (best_goal_node) {
    // Add closest goal node
    std::vector<float> best_goal;
    float min_dist = std::numeric_limits<float>::max();
    for (auto goal : goals) {
      float dist = euclideanDistance({best_goal_node->x, best_goal_node->y}, goal);
      if (dist < min_dist) {
        min_dist = dist;
        best_goal = goal;
      }
    }
    RRTNode::Ptr goal_node =
        std::make_shared<RRTNode>(best_goal[0], best_goal[1],
                                  best_goal_node->g + min_dist, best_goal_node);
    tree.push_back(goal_node);
    std::vector<std::vector<float>> path = reconstructPath_(goal_node);
    clearTree_(tree);
    return simplifyPath(path);
  }

  // std::cout << "No path found after all samples! " << std::endl;
  return {};  // Return an empty path if no valid path found
}