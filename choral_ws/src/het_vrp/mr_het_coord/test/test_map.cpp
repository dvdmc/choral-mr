#include "gtest/gtest.h"
#include "mr_het_coord/map/map.hpp"   // Assuming your class is defined here

class GridMapTest : public ::testing::Test {
protected:
    GridMap* grid_map;
    std::vector<float> origin;
    int width = 10;
    int height = 10;
    float scale = 1.5f;

    void SetUp() override {
        origin = {0.0f, 0.0f};  // Define the origin at the bottom left corner
        grid_map = new GridMap(scale, origin);

        // Setup a simple 10x10 grid with free_thresh of 90 for basic testing
        grid_map->grid_map.resize(height, std::vector<int>(width, 150));  // All cells above free_thresh
        grid_map->FREE_THRESH = 90;

        // Create an occupied area in the center of the grid
        for (int r = height / 3; r < height / 3 + 3; r++)
        {
            for (int c = width / 3; c < width / 3 + 3; c++)
            {
                grid_map->grid_map[r][c] = 0;
            }
        }

        grid_map->printMap();
    }

    void TearDown() override {
        delete grid_map;
    }
};

// Test gridToWorld conversion
TEST_F(GridMapTest, GridToWorldTest) {
    std::vector<float> worldPos = grid_map->gridToWorld(0, 0);
    EXPECT_EQ(worldPos[0], 0.0f);
    EXPECT_EQ(worldPos[1], 13.5f);

    worldPos = grid_map->gridToWorld(5, 4);
    EXPECT_EQ(worldPos[0], 6.0f);
    EXPECT_EQ(worldPos[1], 6.0f);

    worldPos = grid_map->gridToWorld(9, 9);
    EXPECT_EQ(worldPos[0], 13.5f);
    EXPECT_EQ(worldPos[1], 0.0f);  // Top right corner
}

// Test worldToGrid conversion
TEST_F(GridMapTest, WorldToGridTest) {
    std::vector<int> gridPos = grid_map->worldToGrid(0.0f, 0.0f);
    EXPECT_EQ(gridPos[0], 9);
    EXPECT_EQ(gridPos[1], 0);

    gridPos = grid_map->worldToGrid(6.0f, 6.0f);
    EXPECT_EQ(gridPos[0], 5); 
    EXPECT_EQ(gridPos[1], 4);

    gridPos = grid_map->worldToGrid(13.5f, 13.5f);
    EXPECT_EQ(gridPos[0], 0);
    EXPECT_EQ(gridPos[1], 9);
}

// Test isValid function for world coordinates
TEST_F(GridMapTest, IsValidTest) {
    EXPECT_TRUE(grid_map->isValid(0.0f, 0.0f));  // Origin is valid
    EXPECT_FALSE(grid_map->isValid(-1.0f, -1.0f));  // Out of bounds
    EXPECT_FALSE(grid_map->isValid(15.0f, 15.0f));  // Beyond grid
    EXPECT_FALSE(grid_map->isValid(7.5f, 7.5f));  // Beyond grid
}

// Test isVisible function with line of sight
TEST_F(GridMapTest, IsVisibleTest) {
    EXPECT_TRUE(grid_map->isVisible(0.0f, 0.0f, 0.0f, 13.5f, 0.1f));  // Too close increments
    EXPECT_TRUE(grid_map->isVisible(0.0f, 0.0f, 6.0f, 0.0f, 0.1f));  // Too close increments
    EXPECT_FALSE(grid_map->isVisible(0.0f, 0.0f, 13.5f, 13.5f, 0.1f));  // Diagonal across free space
    EXPECT_FALSE(grid_map->isVisible(6.0f, 6.0f, 6.0f, 6.0f, 0.1f));  // Diagonal across free space
}

// Test openFromPGM file loading (Mocking a PGM file content)
// TEST_F(GridMapTest, OpenFromPGMTest) {
//     // Mock file loading (you can test it with a real PGM file separately)
//     EXPECT_TRUE(grid_map->openFromPGM("sample.pgm"));
//     EXPECT_EQ(grid_map->grid_map.size(), height);
//     EXPECT_EQ(grid_map->grid_map[0].size(), width);
// }

// Test computeSDF for computing the Euclidean Signed Distance Field
TEST_F(GridMapTest, ComputeSDFTest) {
    grid_map->computeSDF();
    grid_map->printMap();
    // Check some values in the ESDF for correctness
    // The ESDF should have 0.0 for obstacles and non-negative for free space
    EXPECT_GE(grid_map->sdf_map[0][0], 0.0f);
    EXPECT_LE(grid_map->sdf_map[6][6], 0.0f);  // Should be 0 for obstacle area
}

// Test task generation from free space
TEST_F(GridMapTest, GetTasksTest) {
    auto tasks = grid_map->getTasksFree(1);  // Request tasks with a size of 1
    EXPECT_GT(tasks.size(), 0);  // Tasks should be found
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}