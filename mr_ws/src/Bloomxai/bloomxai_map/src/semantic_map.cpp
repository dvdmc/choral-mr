#include "bloomxai_map/semantic_map.hpp"

#include <eigen3/Eigen/Geometry>
#include <queue>
#include <unordered_set>

namespace Bloomxai {

const int32_t SemanticMap::UnknownProbability = SemanticMap::logods(0.5f);

VoxelGrid<SemCellT>& SemanticMap::grid() {
  return grid_;
}

SemanticMap::SemanticMap(
    double resolution, int sem_dim, std::unique_ptr<BaseSemanticOperator> semantic_operator_)
    : sem_dim_(sem_dim),
      resolution_(resolution),
      grid_(resolution),
      accessor_(grid_.createAccessor()),
      semantic_operator(std::move(semantic_operator_)) {}

const VoxelGrid<SemCellT>& SemanticMap::grid() const {
  return grid_;
}

const SemanticMap::Options& SemanticMap::options() const {
  return options_;
}

void SemanticMap::setOptions(const Options& options) {
  options_ = options;
}

void SemanticMap::addHitPoint(const Vector3D& point, const VSemantics& semantics) {
  const auto coord = grid_.posToCoord(point);
  SemCellT* cell = SemanticMap::ensureCellInitalized(accessor_.value(coord, true));

  // TODO(anonym): Check if only updating once causes artifacts
  if (cell->update_id != _update_count) {
    cell->occ_prob_log =
        std::max(cell->occ_prob_log + options_.prob_hit_log, options_.clamp_min_log);

    semantic_operator->integrateHit(*cell, semantics);

    cell->update_id = _update_count;
    hit_coords_.push_back(coord);
  }
}

void SemanticMap::addMissPoint(const Vector3D& point) {
  const auto coord = grid_.posToCoord(point);
  SemCellT* cell = SemanticMap::ensureCellInitalized(accessor_.value(coord, true));

  if (cell->update_id != _update_count) {
    cell->occ_prob_log =
        std::max(cell->occ_prob_log + options_.prob_miss_log, options_.clamp_min_log);

    semantic_operator->integrateMiss(*cell);

    cell->update_id = _update_count;
    miss_coords_.push_back(coord);
  }
}

bool SemanticMap::isOccupied(const CoordT& coord) const {
  if (auto* cell = accessor_.value(coord, false)) {
    return cell->occ_prob_log > options_.occupancy_threshold_log;
  }
  return false;
}

bool SemanticMap::isUnknown(const CoordT& coord) const {
  if (auto* cell = accessor_.value(coord, false)) {
    return cell->occ_prob_log == options_.occupancy_threshold_log;
  }
  return false;
}

bool SemanticMap::isFree(const CoordT& coord) const {
  if (auto* cell = accessor_.value(coord, false)) {
    return cell->occ_prob_log < options_.occupancy_threshold_log;
  }
  return false;
}

void SemanticMap::updateFreeCells(const Vector3D& origin) {
  auto accessor = grid_.createAccessor();

  // same as addMissPoint, but using lambda will force inlining
  auto clearPoint = [this, &accessor](const CoordT& coord) {
    SemCellT* cell = SemanticMap::ensureCellInitalized(accessor.value(coord, true));
    if (cell->update_id != _update_count) {
      cell->occ_prob_log =
          std::max(cell->occ_prob_log + options_.prob_miss_log, options_.clamp_min_log);

      // Make the semantics more uncertain
      semantic_operator->integrateMiss(*cell);

      cell->update_id = _update_count;
    }
    return true;
  };

  const auto coord_origin = grid_.posToCoord(origin);

  for (const auto& coord_end : hit_coords_) {
    RayIterator(coord_origin, coord_end, clearPoint);
  }
  hit_coords_.clear();

  for (const auto& coord_end : miss_coords_) {
    RayIterator(coord_origin, coord_end, clearPoint);
  }
  miss_coords_.clear();

  if (++_update_count == 4) {
    _update_count = 1;
  }
}

void SemanticMap::getOccupiedVoxels(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    if (cell.occ_prob_log > options_.occupancy_threshold_log) {
      coords.push_back(coord);
    }
  };
  grid_.forEachCell(visitor);
}

void SemanticMap::getMapLimits(std::vector<float>& min, std::vector<float>& max) const {
  min = {
      std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max()};
  max = {
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::min()};

  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    const auto p = grid_.coordToPos(coord);
    if (p.x < min[0])
      min[0] = p.x;
    if (p.x > max[0])
      max[0] = p.x;
    if (p.y < min[1])
      min[1] = p.y;
    if (p.y > max[1])
      max[1] = p.y;
    if (p.z < min[2])
      min[2] = p.z;
    if (p.z > max[2])
      max[2] = p.z;
  };
  grid_.forEachCell(visitor);
}

std::vector<int> SemanticMap::getMapXYSize() const {
  std::vector<float> min, max;
  getMapLimits(min, max);
  return {int((max[0] - min[0]) / resolution_) + 1, int((max[1] - min[1]) / resolution_) + 1};
}

void SemanticMap::getOccupiedVoxelsAndClass(
    std::vector<CoordT>& coords, std::vector<int>& classes) {
  coords.clear();
  classes.clear();
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    if (cell.occ_prob_log > options_.occupancy_threshold_log) {
      coords.push_back(coord);
      classes.push_back(semantic_operator->argmax(cell));
    }
  };
  grid_.forEachCell(visitor);
}

void SemanticMap::getFreeVoxels(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    if (cell.occ_prob_log < options_.occupancy_threshold_log) {
      coords.push_back(coord);
    }
  };
  grid_.forEachCell(visitor);
}

void SemanticMap::getOccupiedVoxelsClassAndSimilarity(
    int query_id, std::vector<CoordT>& coords, std::vector<int>& labels, std::vector<float>& sims) {
  coords.clear();
  labels.clear();
  sims.clear();
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    if (cell.occ_prob_log > options_.occupancy_threshold_log) {
      coords.push_back(coord);
      labels.push_back(semantic_operator->argmax(cell));
      sims.push_back(semantic_operator->getSimilarity(cell, query_id));
    }
  };
  grid_.forEachCell(visitor);
}

void collapseTasks(std::vector<std::vector<int>>& grid, int goal_code = 22) {
  int H = grid.size();
  int W = grid[0].size();
  int min_size = 4;

  std::vector<std::vector<bool>> visited(H, std::vector<bool>(W, false));

  const int dx[8] = {1, -1, 0, 0, 1, -1, 1, -1};
  const int dy[8] = {0, 0, 1, -1, 1, -1, -1, 1};

  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      // Start BFS for each unvisited goal cell
      if (grid[y][x] == goal_code && !visited[y][x]) {
        std::queue<std::pair<int, int>> q;
        std::vector<std::pair<int, int>> region;

        q.push({x, y});
        visited[y][x] = true;

        while (!q.empty()) {
          auto [cx, cy] = q.front();
          q.pop();
          region.push_back({cx, cy});

          // explore 4-neighbors
          for (int k = 0; k < 4; k++) {
            int nx = cx + dx[k];
            int ny = cy + dy[k];

            if (nx >= 0 && nx < W && ny >= 0 && ny < H && !visited[ny][nx] &&
                grid[ny][nx] == goal_code) {
              visited[ny][nx] = true;
              q.push({nx, ny});
            }
          }
        }

        // --- Collapse this region ---
        if (!region.empty()) {
          // find bounding box
          int min_x = W, min_y = H;
          int max_x = 0, max_y = 0;

          for (auto& p : region) {
            min_x = std::min(min_x, p.first);
            min_y = std::min(min_y, p.second);
            max_x = std::max(max_x, p.first);
            max_y = std::max(max_y, p.second);
          }

          // bounding-box sizes (optional)
          int x_size = max_x - min_x + 1;
          int y_size = max_y - min_y + 1;

          // bounding-box center
          int cx = (min_x + max_x) / 2;
          int cy = (min_y + max_y) / 2;

          // region size (optional)
          int region_size = region.size();

          // clear entire region
          for (auto& p : region) {
            grid[p.second][p.first] = 0;
          }

          // mark center cell
          if (x_size >= min_size && y_size >= min_size) {
            grid[cy][cx] = goal_code;
          }
        }
      }
    }
  }
}

std::vector<std::vector<int>> SemanticMap::generate2DGridMap(
    const std::vector<float>& min, const std::vector<float>& max, const std::vector<int>& map_size,
    const std::vector<int>& problematic_classes, const std::vector<int>& task_classes,
    float th_z) const {
  std::vector<std::vector<int>> matrix(map_size[1], std::vector<int>(map_size[0], -1));
  const int FREE_VALUE = 0;
  const int OCCUPIED_VALUE = 100;
  const int PROBLEMATIC_VALUE = 51;
  const int TASK_VALUE = 22;
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    int grid_coord_x = int(coord.x - (min[0] / resolution_));
    int grid_coord_y = int(coord.y - (min[1] / resolution_));
    float coord_z_th = th_z / resolution_;

    bool is_task = false;
    for (int c : task_classes) {
      if (semantic_operator->argmax(cell) == c) {
        is_task = true;
        break;
      }
    }
    bool is_problematic = false;
    if (!is_task) {
      for (int c : problematic_classes) {
        if (semantic_operator->argmax(cell) == c) {
          is_problematic = true;
          break;
        }
      }
    }

    //  TODO: Fix this
    if (grid_coord_y < 0 || grid_coord_y >= map_size[1] || grid_coord_x < 0 ||
        grid_coord_x >= map_size[0]) {
      std::cerr << "Out-of-bounds write: (" << grid_coord_x << ", " << grid_coord_y << ")"
                << std::endl;
      return;  // or continue;
    }

    int prev_value = matrix[grid_coord_y][grid_coord_x];
    if (coord.z > coord_z_th && cell.occ_prob_log > options_.occupancy_threshold_log) {
      matrix[grid_coord_y][grid_coord_x] = OCCUPIED_VALUE;
    } else if (is_problematic && prev_value != OCCUPIED_VALUE) {
      matrix[grid_coord_y][grid_coord_x] = PROBLEMATIC_VALUE;
    } else if (is_task && prev_value != OCCUPIED_VALUE && prev_value != PROBLEMATIC_VALUE) {
      matrix[grid_coord_y][grid_coord_x] = TASK_VALUE;
    } else if (
        prev_value != OCCUPIED_VALUE && prev_value != PROBLEMATIC_VALUE &&
        prev_value != TASK_VALUE) {
      matrix[grid_coord_y][grid_coord_x] = FREE_VALUE;
    }
  };

  grid_.forEachCell(visitor);
  collapseTasks(matrix, TASK_VALUE);
  return matrix;
}

}  // namespace Bloomxai