#ifndef MR_HET_COORD_MAP_HPP
#define MR_HET_COORD_MAP_HPP

#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm>

/// @brief Enumeration for specifying the type of agent
enum AgentType
{
  AERIAL = 0,
  GROUND = 1
};

inline AgentType stringToAgentType(std::string agent_type)
{
  if (agent_type == "AERIAL")
  {
    return AgentType::AERIAL;
  }
  else if (agent_type == "GROUND")
  {
    return AgentType::GROUND;
  }
}

inline std::string agentTypeToString(AgentType agent_type)
{
  if (agent_type == AgentType::AERIAL)
  {
    return "AERIAL";
  }
  else if (agent_type == AgentType::GROUND)
  {
    return "GROUND";
  }
  return "UNKNOWN";
}

class GridMap
{

public:
  // The world positions are measured from the bottom left corner of the map
  int width_px = -1;
  float width_m = -1;
  int height_px = -1;
  float height_m = -1;
  float resolution = 1.0; // resolution for [m/pixel]
  std::vector<float> origin = {0.0, 0.0};
  std::vector<std::vector<int>> grid_map;
  std::vector<std::vector<float>> sdf_map;
  std::vector<std::vector<int>> grid_map_terrain;
  float max_dist = 0.0f;

  int FREE_THRESH = 90;     // if %<90 the cell is free. Values from 0-100
  float DIST_THRESH = 0.5f; // meters
  int TRAVERSABILITY_VALUE = 51;

  GridMap(float const _resolution, std::vector<float> const _origin, float _safety_dist = 0.3f)
      : resolution(_resolution), origin(_origin), DIST_THRESH(_safety_dist) {}

  /**
   * @brief Converts grid coordinates to world coordinates.
   *
   * This function takes the row and column indices of a cell in the
   * grid map and converts them to world coordinates based on the
   * resolution and origin of the map.
   *
   * @param r Row index in the grid map.
   * @param c Column index in the grid map.
   * @return A vector containing the x and y world coordinates.
   */
  std::vector<float> gridToWorld(int r, int c) const
  {
    return {float(c) * resolution + origin[0],
      float(r) * resolution + origin[1]};
  }

  /**
   * @brief Converts world coordinates to grid coordinates.
   *
   * This function takes the x and y world coordinates and converts
   * them to row and column indices in the grid map based on the
   * resolution and origin of the map.
   *
   * @param x World x coordinate.
   * @param y World y coordinate.
   * @return A vector containing the row and column indices in the grid map.
   */
  std::vector<int> worldToGrid(float x, float y) const
  {
    // NOTE: do floor to get the correct grid coordinates
    int c = int(floor((x - origin[0]) / resolution));
    int r = int(floor((y - origin[1]) / resolution));

    return {r, c};
  }

  std::vector<float> getMinXY() const { return gridToWorld(0, 0); }

  std::vector<float> getMaxXY() const
  {
    return gridToWorld(grid_map.size() - 1, grid_map[0].size() - 1);
  }

  /**
   * @brief Checks if a point in the world is valid in the grid map.
   *
   * This function takes the x and y world coordinates and checks if
   * the point is valid in the grid map. A point is valid if it is
   * within the bounds of the grid map and if the corresponding cell
   * in the grid map has a value greater than FREE_THRESH if use_sdf is
   * false, or if the corresponding cell in the signed distance field
   * has a value greater than DIST_THRESH if use_sdf is true.
   *
   * @param x World x coordinate.
   * @param y World y coordinate.
   * @param use_sdf Whether to use the signed distance field or not.
   *                Defaults to false.
   * @return True if the point is valid, false otherwise.
   */
  bool isValid(float x, float y, bool use_sdf = false) const
  {
    // Convert world coordinates to grid coordinates
    std::vector<int> grid = worldToGrid(x, y);
    return grid[0] >= 0 && grid[0] < int(grid_map.size()) && grid[1] >= 0 &&
           grid[1] < int(grid_map[0].size()) && // Check limits
           // Check if the cell is free without SDF
           ((!use_sdf && grid_map[grid[0]][grid[1]] < FREE_THRESH) ||
            // Use SDF to compute dist to obstacles
            (use_sdf && sdf_map[grid[0]][grid[1]] > DIST_THRESH));
  }

  /**
   * @brief Checks if a line segment is visible in the grid map.
   *
   * This function takes the start and end points of a line segment
   * and checks if the line segment is visible in the grid map.
   * A line segment is visible if all the points along the line
   * segment are valid in the grid map, either by being within the
   * bounds of the grid map and having a value greater than FREE_THRESH
   * if use_sdf is false, or if the corresponding cell in the signed
   * distance field has a value greater than DIST_THRESH if use_sdf is
   * true.
   *
   * @param x1 World x coordinate of the start point.
   * @param y1 World y coordinate of the start point.
   * @param x2 World x coordinate of the end point.
   * @param y2 World y coordinate of the end point.
   * @param increment The increment to use when stepping along the line.
   *                  Defaults to 0.001f.
   * @param use_sdf Whether to use the signed distance field or not.
   *                Defaults to false.
   * @return True if the line segment is visible, false otherwise.
   */
  bool isVisible(float x1, float y1, float x2, float y2,
                 float increment = 0.001f, bool use_sdf = false) const
  {
    // Calculate the differences in x and y
    float dx = x2 - x1;
    float dy = y2 - y1;

    // Calculate the distance between the two points
    float distance = std::sqrt(dx * dx + dy * dy);

    // If the points are too close, treat them as visible
    if (distance <= increment)
    {
      return isValid(x1, y1, use_sdf) && isValid(x2, y2, use_sdf);
    }

    // Calculate the number of steps based on the distance
    // int num_steps = int(distance / increment) + 1;

    // Step along the line and check each point
    float t = 0.0f;
    while (t < 1.0f)
    {
      // Linear interpolation between the points
      float x = x1 + t * dx;
      float y = y1 + t * dy;

      // If any point along the line is not valid, return false
      if (!isValid(x, y, use_sdf))
      {
        return false;
      }

      if (!use_sdf)
      {
        t += increment / distance;
      }
      else
      {
        t += sdf_map[worldToGrid(x, y)[0]][worldToGrid(x, y)[1]] / distance;
      }
    }

    // If all points are valid, return true
    return true;
  }

  bool setMap(int const width_px, int const height_px, std::vector<std::vector<int>> const &map)
  {
    this->width_px = width_px;
    this->height_px = height_px;
    this->width_m = width_px * resolution;
    this->height_m = height_px * resolution;
    grid_map = map;
    grid_map_terrain.clear();
    grid_map_terrain.resize(height_px, std::vector<int>(width_px, 0));

    // Transform -1 (unknown) to 100 (occupied)
    for (int i = 0; i < height_px; i++)
    {
      for (int j = 0; j < width_px; j++)
      {
        if (grid_map[i][j] == -1)
        {
          grid_map[i][j] = 100;
        }
      }
    }

    computeTraversability();
    computeSDF();
    return true;
  }

  /**
   * @brief Opens a PGM file and loads it into a GridMap object.
   * @details This function reads the PGM file header, skips any comments,
   * reads the width and height of the image, and reads the pixel data.
   * It then constructs a GridMap object from the pixel data and
   * computes the signed distance field.
   * @param filename The name of the PGM file to open.
   * @return True if the file was opened successfully, false otherwise.
   */
  bool openFromPGM(std::string const &filename)
  {
    std::cout << "Opening file: " << filename << std::endl;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
      std::cerr << "Error opening file: " << filename << std::endl;
      return false;
    }

    // Read PGM header
    std::string magicNumber;
    file >> magicNumber; // Should be "P5"
    if (magicNumber != "P2")
    {
      std::cerr << "Invalid PGM file" << std::endl;
      return false;
    }

    // Skip comments
    std::string comment;
    file >> comment;
    // std::cout << "File comment: " << comment << std::endl;
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // _px and heigh are on the next line
    file >> width_px >> height_px;
    width_m = width_px * resolution;
    height_m = height_px * resolution;
    std::cout << "Width: " << width_px << ", Height: " << height_px
              << std::endl;
    std::cout << "Width m: " << width_m << ", Height m: " << height_m
              << std::endl;
    int maxVal;
    file >> maxVal;

    // Read pixel data and construct the grid map
    grid_map.resize(height_px, std::vector<int>(width_px, 100));
    grid_map_terrain.resize(height_px, std::vector<int>(width_px, 0));

    // From image space to map, axis is inverted NOTE: This should be the only Y axis inversion
    for (int i = height_px - 1; i >= 0; i--) 
    {
      for (int j = 0; j < width_px; j++)
      {
        int pixel;
        file >> pixel;
        // Values in a occupancy gridmap go from 0 to 100 and high means more occ. prob
        // Since we paint free space in white, we need to invert it and make it go from 100 to 0
        grid_map[i][j] = 100 - (pixel * 100) / 255;
        if (pixel == TRAVERSABILITY_VALUE)
        {
          // This indicates bad traversability
          std::cout << "Bad traversability: " << pixel << std::endl;
          grid_map_terrain[i][j] = 1;
        }
      }
    }

    // Close the file
    file.close();
    computeSDF();

    return true;
  }

  /**
    * @brief Computes the traversability of the current grid map.
    * @details This function computes the traversability of the current grid map
    * based on the grid_map it fills the grid_map_terrain.
    */
  void computeTraversability()
  {
    grid_map_terrain.resize(height_px, std::vector<int>(width_px, 0));
    for (int i = 0; i < height_px; i++)
    {
      for (int j = 0; j < width_px; j++)
      {
        if (grid_map[i][j] == TRAVERSABILITY_VALUE)
        {
          // This indicates bad traversability
          grid_map_terrain[i][j] = 1;
        }
      }
    }
  }

  /**
   * @brief Computes the signed distance field (SDF) from the current grid map.
   * @details This function computes the signed distance field (SDF) from the
   * member gridmap, which is a 2D grid where each cell contains the minimum
   * distance from that cell to the closest obstacle. The distance values are
   * in the world resolution.
   */
  void computeSDF()
  {

    std::vector<std::vector<int>> binary_grid(
        grid_map.size(), std::vector<int>(grid_map[0].size()));

    for (size_t i = 0; i < grid_map.size(); ++i)
    {
      for (size_t j = 0; j < grid_map[i].size(); ++j)
      {
        if (grid_map[i][j] < FREE_THRESH)
        {
          binary_grid[i][j] = 0; // Free space
        }
        else
        {
          binary_grid[i][j] = 1; // Obstacle
        }
      }
    }

    std::vector<std::vector<float>> dist_grid(
        grid_map.size(), std::vector<float>(grid_map[0].size()));
    float inf = std::numeric_limits<float>::infinity();

    // Initialize distance grid: 0 for obstacles, inf for free space
    for (size_t i = 0; i < dist_grid.size(); i++)
    {
      for (size_t j = 0; j < dist_grid[i].size(); j++)
      {
        if (binary_grid[i][j] == 1)
        {
          dist_grid[i][j] = 0.0f; // Obstacle
        }
        else
        {
          dist_grid[i][j] = inf; // Free space
        }
      }
    }

    std::queue<std::pair<int, int>> q;

    // Add all the origin (obstacle) points to the queue
    for (size_t i = 0; i < binary_grid.size(); i++)
    {
      for (size_t j = 0; j < binary_grid[i].size(); j++)
      {
        if (binary_grid[i][j] == 1)
        {
          q.push({i, j});
        }
      }
    }

    while (!q.empty())
    {
      std::pair<int, int> point = q.front();
      q.pop();
      int i = point.first;
      int j = point.second;

      // Check the neighbors
      if (i > 0 && dist_grid[i - 1][j] > dist_grid[i][j] + 1)
      {
        dist_grid[i - 1][j] = dist_grid[i][j] + 1.0f;
        q.push({i - 1, j});
      }
      if (i < int(grid_map.size()) - 1 && dist_grid[i + 1][j] > dist_grid[i][j] + 1)
      {
        dist_grid[i + 1][j] = dist_grid[i][j] + 1.0f;
        q.push({i + 1, j});
      }
      if (j > 0 && dist_grid[i][j - 1] > dist_grid[i][j] + 1)
      {
        dist_grid[i][j - 1] = dist_grid[i][j] + 1.0f;
        q.push({i, j - 1});
      }
      if (j < int(grid_map[0].size()) - 1 &&
          dist_grid[i][j + 1] > dist_grid[i][j] + 1)
      {
        dist_grid[i][j + 1] = dist_grid[i][j] + 1.0f;
        q.push({i, j + 1});
      }
    }

    // Transform to real world coordinates
    for (size_t i = 0; i < dist_grid.size(); i++)
    {
      for (size_t j = 0; j < dist_grid[i].size(); j++)
      {
        dist_grid[i][j] = dist_grid[i][j] * resolution;
        if (dist_grid[i][j] > max_dist)
        {
          max_dist = dist_grid[i][j];
        }
      }
    }

    std::cout << "SDF computed!" << std::endl;
    sdf_map = dist_grid;
  }

  /**
   * @brief Returns a vector of points in free space, in the world coordinate system.
   * @details These points are the centers of the cells in the grid map that are
   * considered free space (i.e. the value in the grid map is greater than
   * FREE_THRESH). The points are ordered in a way that considers as first task
   * the cells in the bottom left corner of the map, and then the cells in the
   * @return A vector of points in free space, in the world coordinate system.
   */
  std::vector<std::vector<float>> getFreeSpace() const
  {
    std::vector<std::vector<float>> freeSpace;
    for (int r = int(grid_map.size()) - 1; r >= 0;
         r--) // Gather the tasks the other way around in the vertical axis
    {
      for (size_t c = 0; c < grid_map[0].size(); c++)
      {
        if (grid_map[r][c] < FREE_THRESH)
        {
          std::vector<float> pos_corner = gridToWorld(r, c);
          std::vector<float> pos = {float(pos_corner[0]) + resolution / 2,
                                    pos_corner[1] + resolution / 2};
          freeSpace.push_back({pos[0], pos[1]});
        }
      }
    }

    return freeSpace;
  }

  std::vector<std::vector<float>> getTasksFree(size_t size) const
  {
    std::vector<std::vector<float>> tasks;
    int num_free = 0;
    // Check if there is a space of "size" with at least FREE_THRESH%
    // of free space. The task will be at the center
    for (size_t r = 0; r < grid_map.size() - size;
         r +=
         size) // Gather the tasks the other way around in the vertical axis
    {
      for (size_t c = 0; c < grid_map[0].size() - size; c += size)
      {
        // Check the space to be covered
        for (size_t i = 0; i < size; i++)
        {
          for (size_t j = 0; j < size; j++)
          {
            if (grid_map[r + i][c + j] < FREE_THRESH)
            {
              num_free++;
            }
          }
        }
        if (num_free >= size * size * 0.95)
        {
          tasks.push_back({gridToWorld(r, c)[0] + size * resolution / 2,
                           gridToWorld(r, c)[1] - size * resolution / 2});
        }
        num_free = 0;
      }
    }

    return tasks;
  }

  /**
   * @brief Returns a vector of tasks in the world coordinate system.
   * @details The tasks are computed by finding the positions in the grid map
   * with value 200 (color in the PGM file), which are considered tasks.
   * The task is at the center of the free space.
   * @return A vector of tasks in the world coordinate system.
   */
  std::vector<std::vector<float>> getTasksFromGridMap() const
  {
    std::vector<std::vector<float>> tasks;
    for (size_t r = 0; r < grid_map.size() - 1;
         r++) // Gather the tasks the other way around in the vertical axis
    {
      for (size_t c = 0; c < grid_map[0].size() - 1; c++)
      {
        if (grid_map[r][c] == 22) // 22 = 100 - int(200 * 100 / 255) color is 200 in PGM
        {
          std::vector<float> pos_corner = gridToWorld(r, c);
          std::vector<float> pos = {float(pos_corner[0]) + resolution / 2,
                                    pos_corner[1] + resolution / 2};
          tasks.push_back({pos[0], pos[1]});
        }
      }
    }
    // Invert the order of the tasks
    std::reverse(tasks.begin(), tasks.end());
    return tasks;
  }

  /**
   * @brief Returns the traversability value for a given position and agent type.
   * @details The traversability value is 0.8 if the position has traversability
   * problems for ground robots and 1.0 in the case of not having traversability
   * problems or being an aerial robot.
   * @param x World x coordinate.
   * @param y World y coordinate.
   * @param agent_type Agent type.
   * @return Traversability value.
   */
  float getTraversability(float x, float y, AgentType agent_type) const
  {
    std::vector<int> cell = worldToGrid(x, y);
    if (grid_map_terrain[cell[0]][cell[1]] == 1 && agent_type == AgentType::GROUND)
    {
      return 0.0f;
    }
    else
    {
      return 1.0f;
    }
  }

  /**
   * @brief Returns the distance value for a given position.
   * @param x World x coordinate.
   * @param y World y coordinate.
   * @return Distance value.
   */
  float getDistance(float x, float y) const
  {
    std::vector<int> cell = worldToGrid(x, y);

    return sdf_map[cell[0]][cell[1]];
  }

  void printMap() const
  {
    std::cout << "Height: " << height_px << ", Width: " << width_px
              << std::endl;

    std::cout << "Grid Map:\n";
    for (int r = 0; r < height_px; ++r)
    {
      for (int c = 0; c < width_px; ++c)
      {
        int cell = grid_map[r][c];
        std::cout << " " << cell << " " << std::endl;
      }
      std::cout << "\n";
    }
  }

  void printSdfMap() const
  {
    std::cout << "Height: " << height_px << ", Width: " << width_px 
              << std::endl;

    std::cout << "Grid Map:\n";
    for (int r = 0; r < height_px; ++r)
    {
      for (int c = 0; c < width_px; ++c)
      {
        float cell = sdf_map[r][c];
        std::cout << " " << cell << " ";
      }
      std::cout << "\n";
    }
  }

  void printPath(std::vector<std::vector<float>> const &path) const
  {
    std::vector<std::vector<int>> grid_path;
    for (auto const &p : path)
    {
      grid_path.push_back(worldToGrid(p[0], p[1]));
    }
    for (size_t r = 0; r < grid_map.size(); ++r)
    {
      for (size_t c = 0; c < grid_map[0].size(); ++c)
      {
        bool is_in_path = false;
        int index = 0;
        for (auto const &grid : grid_path)
        {
          if (size_t(grid[0]) == r && size_t(grid[1]) == c)
          {
            is_in_path = true;
            break;
          }
          index++;
        }
        if (is_in_path)
        {
          std::cout << index << " ";
        }
        else
        {
          std::cout << (grid_map[r][c] ? "o" : "x") << " ";
        }
      }
      std::cout << "\n";
    }
  }

  void printAssignment(
      std::vector<std::vector<std::vector<float>>> const &paths) const
  {
    std::vector<std::vector<std::vector<int>>> grid_paths;
    for (auto const &k : paths)
    {
      std::vector<std::vector<int>> grid_path;
      for (auto const &p : k)
      {
        grid_path.push_back(worldToGrid(p[0], p[1]));
      }
      grid_paths.push_back(grid_path);
    }

    for (size_t r = 0; r < grid_map.size(); ++r)
    {
      for (size_t c = 0; c < grid_map[0].size(); ++c)
      {
        bool is_in_path = false;
        int index = 0; // Index for the agent assignment

        for (auto const &grid_path : grid_paths)
        { // For each agent
          for (auto const &grid : grid_path)
          { // For each position
            if (size_t(grid[0]) == r && size_t(grid[1]) == c)
            {
              is_in_path = true;
              break;
            }
          }
          if (is_in_path)
          {
            break;
          }
          index++;
        }

        if (is_in_path)
        {
          std::cout << index << " ";
        }
        else
        {
          std::cout << (grid_map[r][c] ? "o" : "x") << " ";
        }
      }
      std::cout << "\n";
    }

    for (size_t k = 0; k < grid_paths.size(); ++k)
    {
      std::cout << "Agent " << k << ": ";
      for (auto const &grid : grid_paths[k])
      {
        std::cout << "(" << grid[0] << " " << grid[1] << ") ";
      }
      std::cout << "\n";
    }
  }
};

#endif