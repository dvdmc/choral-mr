#!/usr/bin/env python3

import rclpy
from sensors_tools_ros.semantic_ros import SemanticNode

def main(args=None):
    rclpy.init(args=args)

    node = SemanticNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()