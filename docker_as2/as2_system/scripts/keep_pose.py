#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class PoseRepublisher(Node):
    def __init__(self):
        super().__init__('pose_republisher')
        
        # Initialize subscriber
        self.subscription = self.create_subscription(
            PoseStamped,
            '/mavros/vision_pose/pose',
            self.pose_callback,
            10
        )

        # Initialize publisher
        self.publisher = self.create_publisher(
            PoseStamped,
            '/mavros/setpoint_position/local',
            10
        )

        self.current_pose = None
        
        # Create timer for publishing at >2 Hz
        self.timer = self.create_timer(0.1, self.publish_pose)  # ~10 Hz

    def pose_callback(self, msg):
        self.get_logger().debug('Pose received.')
        self.current_pose = msg if self.current_pose is None else self.current_pose

    def publish_pose(self):
        print("Triggered")
        if self.current_pose:
            self.current_pose.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(self.current_pose)
            self.get_logger().debug('Pose published.')

def main(args=None):
    rclpy.init(args=args)
    node = PoseRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
