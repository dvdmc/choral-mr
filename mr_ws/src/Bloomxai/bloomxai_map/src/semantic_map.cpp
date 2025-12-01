#include "bloomxai_map/semantic_map.hpp"

#include <eigen3/Eigen/Geometry>
#include <unordered_set>

namespace Bloomxai {

const int32_t SemanticMap::UnknownProbability = SemanticMap::logods(0.5f);

VoxelGrid<SemanticMap::SemCellT>& SemanticMap::grid() {
  return _grid;
}

SemanticMap::SemanticMap(double resolution, int sem_dim)
    : _sem_dim(sem_dim),
      _resolution(resolution),
      UnknownSemProbs(1.0f / sem_dim),
      UnknownSemLogOdds(SemanticMap::logods(UnknownSemProbs)),
      UnknownLabels(0),
      UnknownFeatures(0),
      _options(sem_dim, UnknownSemProbs),
      _grid(resolution),
      _accessor(_grid.createAccessor()) {}

const VoxelGrid<SemanticMap::SemCellT>& SemanticMap::grid() const {
  return _grid;
}

const SemanticMap::Options& SemanticMap::options() const {
  return _options;
}

void SemanticMap::setOptions(const Options& options) {
  _options = options;
}

void SemanticMap::addHitPoint(const Vector3D& point, const SemanticMap::VSemantics& semantics) {
  const auto coord = _grid.posToCoord(point);
  SemCellT* cell = SemanticMap::ensureCellInitalized(_accessor.value(coord, true));

  // TODO(anonym): Check if only updating once causes artifacts
  if (cell->update_id != _update_count) {
    cell->occ_prob_log =
        std::max(cell->occ_prob_log + _options.prob_hit_log, _options.clamp_min_log);

    integrator_->integrateHit(*cell, semantics);

    cell->update_id = _update_count;
    _hit_coords.push_back(coord);
  }
}

void SemanticMap::addMissPoint(const Vector3D& point) {
  const auto coord = _grid.posToCoord(point);
  SemCellT* cell = SemanticMap::ensureCellInitalized(_accessor.value(coord, true));

  if (cell->update_id != _update_count) {
    cell->occ_prob_log =
        std::max(cell->occ_prob_log + _options.prob_miss_log, _options.clamp_min_log);

    integrator_->integrateMiss(*cell);

    cell->update_id = _update_count;
    _miss_coords.push_back(coord);
  }
}

bool SemanticMap::isOccupied(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->occ_prob_log > _options.occupancy_threshold_log;
  }
  return false;
}

bool SemanticMap::isUnknown(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->occ_prob_log == _options.occupancy_threshold_log;
  }
  return false;
}

bool SemanticMap::isFree(const CoordT& coord) const {
  if (auto* cell = _accessor.value(coord, false)) {
    return cell->occ_prob_log < _options.occupancy_threshold_log;
  }
  return false;
}

void SemanticMap::updateFreeCells(const Vector3D& origin) {
  auto accessor = _grid.createAccessor();

  // same as addMissPoint, but using lambda will force inlining
  auto clearPoint = [this, &accessor](const CoordT& coord) {
    SemCellT* cell = SemanticMap::ensureCellInitalized(accessor.value(coord, true));
    if (cell->update_id != _update_count) {
      cell->occ_prob_log =
          std::max(cell->occ_prob_log + _options.prob_miss_log, _options.clamp_min_log);

      // Make the semantics more uncertain
      cell->sem_prob_log = SemanticMap::vlogods(
          SemanticMap::regularizeSemantic(SemanticMap::vprob(cell->sem_prob_log)));

      cell->update_id = _update_count;
    }
    return true;
  };

  const auto coord_origin = _grid.posToCoord(origin);

  for (const auto& coord_end : _hit_coords) {
    RayIterator(coord_origin, coord_end, clearPoint);
  }
  _hit_coords.clear();

  for (const auto& coord_end : _miss_coords) {
    RayIterator(coord_origin, coord_end, clearPoint);
  }
  _miss_coords.clear();

  if (++_update_count == 4) {
    _update_count = 1;
  }
}

void SemanticMap::getOccupiedVoxels(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    if (cell.occ_prob_log > _options.occupancy_threshold_log) {
      coords.push_back(coord);
    }
  };
  _grid.forEachCell(visitor);
}

void SemanticMap::getMapLimits(std::vector<float>& min, std::vector<float>& max) const {
  min = {
      std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max()};
  max = {
      std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::min()};

  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    const auto p = _grid.coordToPos(coord);
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
  _grid.forEachCell(visitor);
}

std::vector<int> SemanticMap::getMapXYSize() const {
  std::vector<float> min, max;
  getMapLimits(min, max);
  return {int((max[0] - min[0]) / _resolution) + 1, int((max[1] - min[1]) / _resolution) + 1};
}

void SemanticMap::getOccupiedVoxelsAndClass(
    std::vector<CoordT>& coords, std::vector<int>& classes) {
  coords.clear();
  classes.clear();
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    if (cell.occ_prob_log > _options.occupancy_threshold_log) {
      coords.push_back(coord);
      classes.push_back(cell.argmax(cell.sem_prob_log));
    }
  };
  _grid.forEachCell(visitor);
}

void SemanticMap::getFreeVoxels(std::vector<CoordT>& coords) {
  coords.clear();
  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    if (cell.occ_prob_log < _options.occupancy_threshold_log) {
      coords.push_back(coord);
    }
  };
  _grid.forEachCell(visitor);
}

std::vector<std::vector<int>> SemanticMap::generate2DGridMap(
    const std::vector<float>& min, const std::vector<float>& max, const std::vector<int>& map_size,
    const std::vector<int>& problematic_classes, float th_z) const {
  std::vector<std::vector<int>> matrix(map_size[1], std::vector<int>(map_size[0], -1));

  auto visitor = [&](SemCellT& cell, const CoordT& coord) {
    int grid_coord_x = int(coord.x - (min[0] / _resolution));
    int grid_coord_y = int(coord.y - (min[1] / _resolution));
    float coord_z_th = th_z / _resolution;

    bool is_problematic = false;
    for (int c : problematic_classes) {
      if (cell.argmax(cell.sem_prob_log) == c) {
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
    if (coord.z > coord_z_th && cell.occ_prob_log > _options.occupancy_threshold_log) {
      matrix[grid_coord_y][grid_coord_x] = 100;
    } else if (is_problematic && prev_value != 100) {
      matrix[grid_coord_y][grid_coord_x] = 51;
    } else if (prev_value != 100 && prev_value != 51) {
      matrix[grid_coord_y][grid_coord_x] = 0;
    }
  };

  _grid.forEachCell(visitor);
  return matrix;
}

}  // namespace Bloomxai