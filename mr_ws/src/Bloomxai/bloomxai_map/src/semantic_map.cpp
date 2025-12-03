#include "bloomxai_map/semantic_map.hpp"

#include <eigen3/Eigen/Geometry>
#include <unordered_set>

namespace Bloomxai {

const int32_t SemanticMap::UnknownProbability = SemanticMap::logods(0.5f);

VoxelGrid<SemCellT>& SemanticMap::grid() {
  return grid_;
}

SemanticMap::SemanticMap(double resolution, int sem_dim, std::unique_ptr<BaseSemanticOperator> semantic_operator_)
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

void SemanticMap::getOccupiedVoxelsAndSimilarity(int query_id, std::vector<CoordT>& coords, std::vector<float>& similarities) {
  coords.clear();
  similarities.clear();
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    if (cell.occ_prob_log > options_.occupancy_threshold_log) {
      coords.push_back(coord);
      similarities.push_back(semantic_operator->getSimilarity(cell, query_id));
    }
  };
  grid_.forEachCell(visitor);
}

std::vector<std::vector<int>> SemanticMap::generate2DGridMap(
    const std::vector<float>& min, const std::vector<float>& max, const std::vector<int>& map_size,
    const std::vector<int>& problematic_classes, float th_z) const {
  std::vector<std::vector<int>> matrix(map_size[1], std::vector<int>(map_size[0], -1));

  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    int grid_coord_x = int(coord.x - (min[0] / resolution_));
    int grid_coord_y = int(coord.y - (min[1] / resolution_));
    float coord_z_th = th_z / resolution_;

    bool is_problematic = false;
    for (int c : problematic_classes) {
      if (semantic_operator->argmax(cell) == c) {
        is_problematic = true;
        break;
      }
    }
    //  std::cout << "Coord: " << coord.x << "," << coord.y << "," << coord.z << " grid: " <<
    //  grid_coord_x
    //           << "," << grid_coord_y << " size: " << map_size[0] << "," << map_size[1] <<
    //           std::endl;
    // std::cout << "Coord Z: " << coord.z << " coord_z_th: " << coord_z_th << std::endl;

    // std::cout << "Map size: " << map_size[0] << ", " << map_size[1]
    //           << " Matrix size: " << matrix.size() << ", " << matrix[0].size() << std::endl;
    if (grid_coord_y < 0 || grid_coord_y >= map_size[1] || grid_coord_x < 0 ||
        grid_coord_x >= map_size[0]) {
      std::cerr << "Out-of-bounds write: (" << grid_coord_x << ", " << grid_coord_y << ")"
                << std::endl;
      return;  // or continue;
    }

    int prev_value = matrix[grid_coord_y][grid_coord_x];
    if (coord.z > coord_z_th && cell.occ_prob_log > options_.occupancy_threshold_log) {
      matrix[grid_coord_y][grid_coord_x] = 100;
    } else if (is_problematic && prev_value != 100) {
      matrix[grid_coord_y][grid_coord_x] = 51;
    } else if (prev_value != 100 && prev_value != 51) {
      matrix[grid_coord_y][grid_coord_x] = 0;
    }
  };

  grid_.forEachCell(visitor);
  return matrix;
}

}  // namespace Bloomxai