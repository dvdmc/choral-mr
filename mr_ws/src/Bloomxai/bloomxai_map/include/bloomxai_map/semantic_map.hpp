#pragma once
#include <any>
#include <bonxai/bonxai.hpp>
#include <bonxai/serialization.hpp>
#include <eigen3/Eigen/Geometry>
#include <iostream>

namespace Bloomxai {
using namespace Bonxai;

using VSemantics = Eigen::VectorXf;

template <class Functor>
void RayIterator(const CoordT& key_origin, const CoordT& key_end, const Functor& func);

inline void ComputeRay(const CoordT& key_origin, const CoordT& key_end, std::vector<CoordT>& ray) {
  ray.clear();
  RayIterator(key_origin, key_end, [&ray](const CoordT& coord) {
    ray.push_back(coord);
    return true;
  });
}

struct SemCellT {
  int32_t update_id : 4;
  int32_t occ_prob_log : 28;
  int32_t sem_dim;
  int32_t count = 0;
  VSemantics semantics;  // probability vector, features, etc.

  SemCellT()
      : update_id(0),
        occ_prob_log(-1),
        sem_dim(0),
        count(0) {}

  // // Calculate serialized size in bytes for serialization
  // [[nodiscard]] size_t size() const {
  //   if (semantics.size() == 0) {
  //     throw std::runtime_error("Used size() on an empty SemCellT.");
  //   }
  //   return sizeof(int32_t)                        // update_id + occ_prob_log (packed into 4
  //   bytes)
  //          + sizeof(int32_t)                      // sem_dim (4 bytes)
  //          + (semantics.size() * sizeof(float));  // semantics data
  // }
};

class BaseSemanticOperator {
 public:
  int sem_dim;
  int queries_size = 0;

  BaseSemanticOperator(int sem_dim)
      : sem_dim(sem_dim) {}
  virtual ~BaseSemanticOperator() = default;

  virtual void setOptions(const std::any& options) = 0;

  // Initialize a new cell (prior)
  virtual void initialize(SemCellT& cell) const = 0;

  // Update a cell given a hit measurement
  virtual void integrateHit(SemCellT& cell, const VSemantics& measurement) const = 0;

  // Update a cell given a miss (ray passing through free space)
  virtual void integrateMiss(SemCellT& cell) const = 0;

  virtual int argmax(const SemCellT& cell) const = 0;

  virtual bool isReady() const = 0;

  // No similarity concept
  virtual float getSimilarity(const SemCellT& cell, int query_id) const {
    return 0.0f;
  };
};

/**
 * @brief The SemanticMap works similarly to ProbabilisticMap but includes
 * different fusion methods for semantic information. Semantics might be:
 * probability vector for classes, features, or labels.
 */
class SemanticMap {
 private:
  int sem_dim_;
  double resolution_;

 public:
  using Vector3D = Eigen::Vector3d;

  static constexpr float kFixedPrecision = 1e6f;
  static constexpr float kEps = 1e-9f;

  std::unique_ptr<BaseSemanticOperator> semantic_operator;

  /// Compute the logds, but return the result as an integer,
  /// The real number is represented as a fixed precision
  /// integer (6 decimals after the comma)
  [[nodiscard]] static constexpr int32_t logods(float prob) {
    return int32_t(kFixedPrecision * std::log((prob + kEps) / (1.0 - prob + kEps)));
  }

  /// Expect the fixed comma value returned by logods()
  [[nodiscard]] static constexpr float prob(int32_t logods_fixed) {
    float logods = float(logods_fixed) * (1.0 / kFixedPrecision);
    return (1.0 - 1.0 / (1.0 + std::exp(logods)));
  }

  static const int32_t UnknownProbability;

  SemCellT* ensureCellInitalized(SemCellT* cell) {
    if (cell->semantics.size() == 0) {
      // Uninformed prior initialization
      semantic_operator->initialize(*cell);
    }
    return cell;
  }

  struct Options {
    // Occupancy
    int32_t prob_miss_log = logods(0.4f);
    int32_t prob_hit_log = logods(0.7f);

    int32_t clamp_min_log = logods(0.12f);
    int32_t clamp_max_log = logods(0.97f);

    int32_t occupancy_threshold_log = logods(0.5f);

    Options() {}
  };

  SemanticMap(
      double resolution, int sem_dim, std::unique_ptr<BaseSemanticOperator> semantic_operator);

  [[nodiscard]] VoxelGrid<SemCellT>& grid();

  [[nodiscard]] const VoxelGrid<SemCellT>& grid() const;

  [[nodiscard]] const Options& options() const;

  void setOptions(const Options& options);

  /**
   * @brief insertPointCloud will update the probability map
   * with a new set of detections.
   * The template function can accept points of different types,
   * such as pcl:Point, Eigen::Vector or Bonxai::Point3d
   *
   * Both origin and points must be in world coordinates
   *
   * @param points   a vector of points with associated semantic measurements
   * @param semantics a vector of semantic measurements
   * @param origin   origin of the point cloud
   * @param max_range max range of the ray, if exceeded, we will use that to
   * compute a free space
   */
  template <typename PointT, typename AllocatorP, typename AllocatorSem>
  void insertPointCloud(
      const std::vector<PointT, AllocatorP>& points,
      const std::vector<VSemantics, AllocatorSem>& semantics, const PointT& origin,
      double max_range);

  // This function is usually called by insertPointCloud
  // We expose it here to add more control to the user.
  // Once finished adding points, you must call updateFreeCells()
  void addHitPoint(const Vector3D& point, const VSemantics& semantics);

  // This function is usually called by insertPointCloud
  // We expose it here to add more control to the user.
  // Once finished adding points, you must call updateFreeCells()
  void addMissPoint(const Vector3D& point);

  [[nodiscard]] bool isOccupied(const Bonxai::CoordT& coord) const;

  [[nodiscard]] bool isUnknown(const Bonxai::CoordT& coord) const;

  [[nodiscard]] bool isFree(const Bonxai::CoordT& coord) const;

  double getResolution() const {
    return resolution_;
  }

  void getOccupiedVoxels(std::vector<Bonxai::CoordT>& coords);

  void getOccupiedVoxelsAndClass(std::vector<Bonxai::CoordT>& coords, std::vector<int>& classes);

  void getFreeVoxels(std::vector<Bonxai::CoordT>& coords);

  void serializeToFile(const std::string& filename) const;

  void deserializeFromFile(const std::string& filename);

  std::vector<std::vector<int>> generate2DGridMap(
      const std::vector<float>& min, const std::vector<float>& max,
      const std::vector<int>& map_size, const std::vector<int>& problematic_classes,
      const std::vector<int>& task_classes,
      float th_z) const;

  void collapseTasks(std::vector<std::vector<int>>& grid,
                                int goal_code,
                                float radius_cells,
                                int unknown_code,
                                float obstacle_radius_cells) const;


  template <typename PointT>
  void getOccupiedVoxels(std::vector<PointT>& points) {
    thread_local std::vector<Bonxai::CoordT> coords;
    coords.clear();
    getOccupiedVoxels(coords);
    for (const auto& coord : coords) {
      const auto p = grid_.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
  }

  void getMapLimits(std::vector<float>& min, std::vector<float>& max) const;

  std::vector<int> getMapXYSize() const;

  template <typename PointT>
  void getOccupiedVoxelsAndClass(std::vector<PointT>& points, std::vector<int>& point_labels) {
    thread_local std::vector<Bonxai::CoordT> coords;
    thread_local std::vector<int> labels;
    coords.clear();
    labels.clear();
    getOccupiedVoxelsAndClass(coords, labels);
    for (size_t i = 0; i < coords.size(); i++) {
      const auto coord = coords[i];
      const auto p = grid_.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
      point_labels.emplace_back(labels[i]);
    }
  }

  template <typename PointT>
  void getFreeVoxels(std::vector<PointT>& points) {
    thread_local std::vector<Bonxai::CoordT> coords;
    coords.clear();
    getFreeVoxels(coords);
    for (const auto& coord : coords) {
      const auto p = grid_.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
    }
  }

  void getOccupiedVoxelsClassAndSimilarity(
      int query_id, std::vector<Bonxai::CoordT>& coords, std::vector<int>& labels,
      std::vector<float>& sim);

  template <typename PointT>
  void getOccupiedVoxelsClassAndSimilarity(
      int query_id, std::vector<PointT>& points, std::vector<int>& classes,
      std::vector<float>& similarities) {
    thread_local std::vector<Bonxai::CoordT> coords;
    thread_local std::vector<int> labels;
    thread_local std::vector<float> sim;
    coords.clear();
    labels.clear();
    sim.clear();
    getOccupiedVoxelsClassAndSimilarity(query_id, coords, labels, sim);
    for (size_t i = 0; i < coords.size(); i++) {
      const auto coord = coords[i];
      const auto p = grid_.coordToPos(coord);
      points.emplace_back(p.x, p.y, p.z);
      classes.emplace_back(labels[i]);
      similarities.emplace_back(sim[i]);
    }
  }

 private:
  Options options_;
  VoxelGrid<SemCellT> grid_;
  uint8_t _update_count = 1;

  std::vector<CoordT> miss_coords_;
  std::vector<CoordT> hit_coords_;

  mutable Bonxai::VoxelGrid<SemCellT>::Accessor accessor_;

  void updateFreeCells(const Vector3D& origin);

  // We reimplement the serialization functions to use SemCellT dynamic size

  inline void Serialize(std::ofstream& out, const VoxelGrid<SemCellT>& grid) const;

  inline VoxelGrid<SemCellT> Deserialize(
      std::istream& input, const HeaderInfo& info, size_t sem_dim);

  void WriteSemCellT(std::ostream& out, const SemCellT& cell) const {
    // Pack the bitfield
    int32_t packed = ((cell.update_id & 0xF) << 28) | (cell.occ_prob_log & 0x0FFFFFFF);
    out.write(reinterpret_cast<const char*>(&packed), sizeof(int32_t));

    // Write sem_dim
    int32_t sem_dim = cell.semantics.size();
    out.write(reinterpret_cast<const char*>(&sem_dim), sizeof(int32_t));

    // Write count
    int32_t count = cell.semantics.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(int32_t));

    // Write semantics data
    out.write(reinterpret_cast<const char*>(cell.semantics.data()), sizeof(float) * sem_dim);
  }

  SemCellT ReadSemCellT(std::istream& input, int sem_dim) {
    SemCellT out;

    // Read packed int (update_id and occ_prob_log)
    int32_t packed;
    input.read(reinterpret_cast<char*>(&packed), sizeof(int32_t));
    out.update_id = (packed >> 28) & 0xF;
    out.occ_prob_log = packed & 0x0FFFFFFF;

    // Read and check sem_dim from file
    int32_t readsem_dim_;
    input.read(reinterpret_cast<char*>(&readsem_dim_), sizeof(int32_t));

    if (readsem_dim_ != sem_dim) {
      throw std::runtime_error("sem_dim in file does not match expected sem_dim");
    }

    out.sem_dim = readsem_dim_;

    // Read count
    int32_t count;
    input.read(reinterpret_cast<char*>(&count), sizeof(int32_t));

    // Read vector
    out.semantics.resize(sem_dim);
    input.read(reinterpret_cast<char*>(out.semantics.data()), sizeof(float) * sem_dim);

    return out;
  }
};

//--------------------------------------------------

inline void SemanticMap::Serialize(std::ofstream& out, const VoxelGrid<SemCellT>& grid) const {
  char header[256];
  std::string type_name = details::demangle(typeid(SemCellT).name());

  sprintf(
      header, "Bonxai::VoxelGrid<%s,%d,%d>(%lf)\n", type_name.c_str(), grid.innetBits(),
      grid.leafBits(), grid.voxelSize());

  out.write(header, std::strlen(header));

  //------------
  Write(out, uint32_t(grid.rootMap().size()));

  for (const auto& it : grid.rootMap()) {
    const CoordT& root_coord = it.first;
    Write(out, root_coord.x);
    Write(out, root_coord.y);
    Write(out, root_coord.z);

    const auto& innergrid_ = it.second;
    for (size_t w = 0; w < innergrid_.mask().wordCount(); w++) {
      Write(out, innergrid_.mask().getWord(w));
    }
    for (auto inner = innergrid_.mask().beginOn(); inner; ++inner) {
      const uint32_t inner_index = *inner;
      const auto& leafgrid_ = *(innergrid_.cell(inner_index));

      for (size_t w = 0; w < leafgrid_.mask().wordCount(); w++) {
        Write(out, leafgrid_.mask().getWord(w));
      }
      for (auto leaf = leafgrid_.mask().beginOn(); leaf; ++leaf) {
        const uint32_t leaf_index = *leaf;
        const auto& cell = leafgrid_.cell(leaf_index);
        WriteSemCellT(out, cell);  // or any function that writes SemCellT field-by-field
      }
    }
  }
}

inline void SemanticMap::serializeToFile(const std::string& filename) const {
  std::ofstream out(filename, std::ios::binary);
  Bloomxai::SemanticMap::Serialize(out, grid_);
  out.close();
}

inline VoxelGrid<SemCellT> SemanticMap::Deserialize(
    std::istream& input, const HeaderInfo& info, size_t sem_dim) {
  std::string type_name = details::demangle(typeid(SemCellT).name());

  VoxelGrid<SemCellT> grid(info.resolution, info.inner_bits, info.leaf_bits);
  uint32_t root_count = Read<uint32_t>(input);

  for (size_t r = 0; r < root_count; ++r) {
    CoordT root_coord{Read<int32_t>(input), Read<int32_t>(input), Read<int32_t>(input)};

    auto inner_it = grid.rootMap().try_emplace(root_coord, info.inner_bits).first;
    auto& innergrid_ = inner_it->second;

    for (size_t w = 0; w < innergrid_.mask().wordCount(); ++w) {
      innergrid_.mask().setWord(w, Read<uint64_t>(input));
    }

    for (auto inner = innergrid_.mask().beginOn(); inner; ++inner) {
      auto& leafgrid_ = innergrid_.cell(*inner);
      leafgrid_ = grid.allocateLeafGrid();

      for (size_t w = 0; w < leafgrid_->mask().wordCount(); ++w) {
        leafgrid_->mask().setWord(w, Read<uint64_t>(input));
      }

      for (auto leaf = leafgrid_->mask().beginOn(); leaf; ++leaf) {
        uint32_t leaf_index = *leaf;

        leafgrid_->cell(leaf_index) = ReadSemCellT(input, sem_dim);
      }
    }
  }

  return grid;
}

inline void SemanticMap::deserializeFromFile(const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  char header[256];
  in.getline(header, 256);
  Bloomxai::HeaderInfo info = Bloomxai::GetHeaderInfo(header);
  auto newgrid_ = Bloomxai::SemanticMap::Deserialize(in, info, sem_dim_);
  grid_ = std::move(newgrid_);
}

template <typename PointT, typename AllocatorP, typename AllocatorSem>
inline void SemanticMap::insertPointCloud(
    const std::vector<PointT, AllocatorP>& points,
    const std::vector<VSemantics, AllocatorSem>& semantics, const PointT& origin,
    double max_range) {
  const auto from = ConvertPoint<Vector3D>(origin);
  const double max_range_sqr = max_range * max_range;

  for (size_t i = 0; i < points.size(); ++i) {
    const auto& point = points[i];
    const auto& semantic = semantics[i];
    const auto to = ConvertPoint<Vector3D>(point);
    Vector3D vect(to - from);
    const double squared_norm = vect.squaredNorm();

    // Points that exceed the max_range will create a cleaning ray
    if (squared_norm >= max_range_sqr) {
      // The new point will have distance == max_range from origin
      const Vector3D new_point = from + ((vect / std::sqrt(squared_norm)) * max_range);
      addMissPoint(new_point);
    } else {
      addHitPoint(to, semantic);
    }
  }
  updateFreeCells(from);
}

template <class Functor>
inline void RayIterator(const CoordT& key_origin, const CoordT& key_end, const Functor& func) {
  if (key_origin == key_end) {
    return;
  }
  if (!func(key_origin)) {
    return;
  }

  CoordT error = {0, 0, 0};
  CoordT coord = key_origin;
  CoordT delta = (key_end - coord);
  const CoordT step = {delta.x < 0 ? -1 : 1, delta.y < 0 ? -1 : 1, delta.z < 0 ? -1 : 1};

  delta = {
      delta.x < 0 ? -delta.x : delta.x, delta.y < 0 ? -delta.y : delta.y,
      delta.z < 0 ? -delta.z : delta.z};

  const int max = std::max(std::max(delta.x, delta.y), delta.z);

  // maximum change of any coordinate
  for (int i = 0; i < max - 1; ++i) {
    // update errors
    error = error + delta;
    // manual loop unrolling
    if ((error.x << 1) >= max) {
      coord.x += step.x;
      error.x -= max;
    }
    if ((error.y << 1) >= max) {
      coord.y += step.y;
      error.y -= max;
    }
    if ((error.z << 1) >= max) {
      coord.z += step.z;
      error.z -= max;
    }
    if (!func(coord)) {
      return;
    }
  }
}

class ProbabilitySemanticOperator : public BaseSemanticOperator {
 public:
  struct ProbOptions {
    // Larger means more regularization
    float alpha_reg = 0.0f;

    float clamp_min_prob = 0.12f;
    float clamp_max_prob = 0.97f;

    ProbOptions() {}
  };

 private:
  const VSemantics reg_probs_;
  ProbOptions options_;

 public:
  ProbabilitySemanticOperator(int sem_dim)
      : BaseSemanticOperator(sem_dim),
        reg_probs_(VSemantics::Constant(sem_dim, 1.0f / sem_dim)) {};

  virtual void setOptions(const std::any& options) override {
    if (!options.has_value())
      return;
    try {
      const ProbOptions& opt = std::any_cast<const ProbOptions&>(options);
      options_ = opt;
    } catch (const std::bad_any_cast& e) {
      std::cout << "Invalid options type for ProbabilitySemanticOperator" << std::endl;
      return;
    }
  }

  virtual void initialize(SemCellT& cell) const override {
    cell.sem_dim = sem_dim;
    cell.count = 0;
    cell.semantics.resize(sem_dim);
    cell.semantics.setConstant(1.0f / sem_dim);
  }

  virtual void integrateHit(SemCellT& cell, const VSemantics& measurement) const override {
    VSemantics regularized = regularizeSemantics(measurement);

    // Correct element-wise multiply
    cell.semantics = cell.semantics.cwiseProduct(regularized);

    // Correct clamping
    cell.semantics =
        cell.semantics.cwiseMax(options_.clamp_min_prob).cwiseMin(options_.clamp_max_prob);

    // Normalize
    cell.semantics /= cell.semantics.sum();
  }

  virtual void integrateMiss(SemCellT& cell) const override {
    // Add uncertainty to the current semantics
    VSemantics regularized = regularizeSemantics(cell.semantics);

    // Correct element-wise multiply
    cell.semantics = cell.semantics.cwiseProduct(regularized);

    // Correct clamping
    cell.semantics =
        cell.semantics.cwiseMax(options_.clamp_min_prob).cwiseMin(options_.clamp_max_prob);

    // Normalize
    cell.semantics /= cell.semantics.sum();
  }

  VSemantics regularizeSemantics(const VSemantics& semantics) const {
    return (1 - options_.alpha_reg) * semantics + options_.alpha_reg * reg_probs_;
  }

  int argmax(const SemCellT& cell) const override {
    if (cell.semantics.size() == 0)
      return 0;  // Handle empty case
    float maxVal = cell.semantics[0];
    int maxIndex = 0;
    for (int i = 1; i < cell.semantics.size(); i++) {
      if (cell.semantics[i] > maxVal) {
        maxVal = cell.semantics[i];
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  bool isReady() const override {
    return true;
  }
};

class FeatureSemanticOperator : public BaseSemanticOperator {
 public:
  struct FeatOptions {
    // Occupancy threshold to initialize the vector and remove it to save space
    int occ_thres_logods = SemanticMap::logods(0.5f);

    int num_queries = 0;
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> query_embeddings = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>::Zero(0, 0);
  };

 private:
  FeatOptions options_;

 public:
  FeatureSemanticOperator(int sem_dim)
      : BaseSemanticOperator(sem_dim) {}  // N=0 initially

  virtual void setOptions(const std::any& options) override {
    if (!options.has_value())
      return;
    try {
      const FeatOptions& opt = std::any_cast<const FeatOptions&>(options);
      options_ = opt;
      queries_size = opt.num_queries;
    } catch (const std::bad_any_cast& e) {
      std::cout << "Invalid options type for FeatureSemanticOperator" << std::endl;
      return;
    }
  }

  void initialize(SemCellT& cell) const override {
    cell.sem_dim = sem_dim;
    cell.count = 0;
    cell.semantics = VSemantics::Zero(sem_dim);
  }

  void integrateHit(SemCellT& cell, const VSemantics& measurement) const override {
    cell.count++;
    float inv_count = 1.0f / cell.count;
    cell.semantics += inv_count * (measurement - cell.semantics);
  }

  void integrateMiss(SemCellT& cell) const override {
  }

  int argmax(const SemCellT& cell) const override {
    if (options_.query_embeddings.rows() == 0) {
      std::cerr << "Query embeddings are empty\n";
      return 0;
    }
    int max_id = 0;
    float max_sim = 0.0f;
    for (int i = 0; i < options_.num_queries; i++) {
      float similarity = getSimilarity(cell, i);
      if (similarity > max_sim) {
        max_sim = similarity;
        max_id = i;
      }
    }
    return max_id;
  }

  float getSimilarity(const SemCellT& cell, int query_id) const {
    if (options_.query_embeddings.rows() == 0) {
      std::cerr << "Query embeddings are empty\n";
      return 0.0f;
    }
    if (query_id < 0 || query_id >= options_.query_embeddings.rows()) {
      std::cerr << "query_id out of range\n";
      return 0.0f;
    }

    Eigen::RowVectorXf q = options_.query_embeddings.row(query_id);

    const Eigen::VectorXf& c = cell.semantics;

    float q_norm = q.norm();
    float c_norm = c.norm();

    if (q_norm == 0.0f || c_norm == 0.0f)
      return 0.0f;

    return q.dot(c) / (q_norm * c_norm);
  }

  bool isReady() const override {
    return options_.num_queries > 0 && options_.query_embeddings.rows() > 0;
  }
};

}  // namespace Bloomxai
