#include "mr_het_coord_ros/mr_het_map_node.hpp"

MRMapNode::MRMapNode() : Node("mr_map_node"), updated_(false)
{
    readROSParameters();

    pub_map_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("grid_map_out", 100);
    
    if (load_map_from_file_)
    {
        RCLCPP_INFO(this->get_logger(), "Loading map from file: %s", map_filename_.c_str());
        map_ = std::make_shared<GridMap>(map_resolution_, map_center_);
        if (map_->openFromPGM(map_filename_))
        {
            RCLCPP_INFO(this->get_logger(), "Map size: %dx%d", 
            map_->width_px, map_->height_px);
        }
        updated_ = true;
    }
    else
    {
        sub_map_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "grid_map_in", 100, std::bind(&MRMapNode::mapCallback, this, std::placeholders::_1));
            while (map_ == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Waiting for map to be published on %s", sub_map_->get_topic_name());
                rclcpp::spin_some(this->get_node_base_interface());
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
    }
    pub_timer_ = this->create_wall_timer(
        std::chrono::seconds(1), std::bind(&MRMapNode::pubTimerCallback, this));
}

MRMapNode::~MRMapNode() {}

void MRMapNode::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
    map_resolution_ = msg->info.resolution;
    map_center_ = std::vector<float>{float(msg->info.origin.position.x), float(msg->info.origin.position.y)};
    map_ = std::make_shared<GridMap>(map_resolution_, map_center_);
    // Build a matrix from the data
    std::vector<std::vector<int>> grid_map(msg->info.height, std::vector<int>(msg->info.width, 0));
    for (size_t i = 0; i < msg->data.size(); i++)
    {
        grid_map[i / msg->info.width][i % msg->info.width] = msg->data[i];
    }
    map_->setMap(msg->info.width, msg->info.height, grid_map);

    updated_ = true;
}

void MRMapNode::readROSParameters()
{

    this->declare_parameter("load_map_from_file", true);
    this->declare_parameter("pub_updates_only", true);
    this->declare_parameter("map_filename", "");
    this->declare_parameter("map_name", "no_named_map");
    this->declare_parameter("map_resolution", 0.0);
    this->declare_parameter("map_center", std::vector<double>({}));

    this->get_parameter("load_map_from_file", load_map_from_file_);
    this->get_parameter("pub_updates_only", pub_updates_only_);
    if (load_map_from_file_)
    {
        this->get_parameter("map_filename", map_filename_);
        this->get_parameter("map_resolution", map_resolution_);
        std::vector<double> temp;
        this->get_parameter("map_center", temp);
        map_center_ = std::vector<float>(temp.begin(), temp.end());
    }
    this->get_parameter("map_name", map_name_);
}

void MRMapNode::pubTimerCallback()
{
    if (pub_updates_only_ && !updated_)
    {
        return;
    }
    updated_ = false;
    nav_msgs::msg::OccupancyGrid msg = createMapMsg(*map_, this->now());
    pub_map_->publish(msg);
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MRMapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
