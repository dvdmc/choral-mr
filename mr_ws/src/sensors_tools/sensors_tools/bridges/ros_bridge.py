from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import cv2

import rclpy
from rclpy.node import Node
from sensors_tools.base.cameras import CameraData
import tf2_ros
import cv_bridge
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CameraInfo
from rclpy.qos import qos_profile_sensor_data
import message_filters
from scipy.ndimage import median_filter
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation

from sensors_tools.bridges.base_bridge import BaseBridge, BaseBridgeConfig

ROSSensorDataTypes = Literal["rgb", "depth", "pose"]
"""
    List of sensor data to query.
    - "pose": query poses.
    - "rgb": query rgb images.
    - "depth": query depth images.
"""


@dataclass
class ROSBridgeConfig(BaseBridgeConfig):
    """
    Configuration class for AirsimBridge
    """

    data_types: List[ROSSensorDataTypes] = field(
        default_factory=list, metadata={"default": ["rgb", "depth", "pose"]}
    )
    """ Data types to query """

    rgb_topic: str = "/camera/rgb/image_raw"
    """ RGB topic """

    camera_info_topic: str = "/camera/rgb/camera_info"
    """ Camera info topic """

    depth_topic: str = "/camera/aligned_depth_to_color/image_raw"
    """ Depth topic """

    origin_tf: str = "map"
    """ Origin frame to query """

    poses_tf: str = "camera_link"
    """ Poses frame to query """

    node: Optional[Node] = None
    """ ROS node that instantiates the bridge. NOTE: It should be passed to cfg manually since there is no default! """


class ROSBridge(BaseBridge):
    """
    Bridge for ROS
    """

    def __init__(self, cfg: ROSBridgeConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.node = cfg.node
        if self.node is None:
            raise RuntimeError(
                "ROSBridge requires a rclpy.node.Node instance in cfg.node"
            )
        self.cv_bridge = cv_bridge.CvBridge()
        self._ready = False

        self.setup()

    @property
    def ready(self):
        return self._ready or self.check_ready()

    def setup(self):
        """
        Setup the bridge
        """
        # Members
        self.rgb = None
        self.rgb_timestamp = None
        self.depth = None
        self.depth_timestamp = None
        self.pose = None
        self.semantic_gt = None

        self.has_camera_info = False
        self.has_depth_camera_info = False

        if "pose" in self.cfg.data_types:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

        # RELEVANT CAMERA DATA
        if "rgb" in self.cfg.data_types:
            self.node.create_subscription(
                CameraInfo,
                self.cfg.camera_info_topic,
                self.camera_info_callback,
                qos_profile_sensor_data,
            )

        if "depth" in self.cfg.data_types:
            self.node.create_subscription(
                CameraInfo,
                self.cfg.camera_info_topic,
                self.depth_camera_info_callback,
                qos_profile_sensor_data,
            )

        if "rgb" in self.cfg.data_types and "depth" in self.cfg.data_types:
            self.rgb_sub = message_filters.Subscriber(
                self.node,
                RosImage,
                self.cfg.rgb_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self.depth_sub = message_filters.Subscriber(
                self.node,
                RosImage,
                self.cfg.depth_topic,
                qos_profile=qos_profile_sensor_data,
            )
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub], 10, 1.0 # TODO: Check time needed to sync
            )
            self.sync.registerCallback(self.sync_callback)
        else:
            if "rgb" in self.cfg.data_types:
                self.node.create_subscription(
                    RosImage,
                    self.cfg.rgb_topic,
                    self.rgb_callback,
                    qos_profile_sensor_data,
                )
            if "depth" in self.cfg.data_types:
                self.node.create_subscription(
                    RosImage,
                    self.cfg.depth_topic,
                    self.depth_callback,
                    qos_profile_sensor_data,
                )

    def check_ready(self):
        if "rgb" in self.cfg.data_types and (
            self.rgb is None or not self.has_camera_info
        ):
            self.node.get_logger().warn(
                f"RGB data not available yet in {self.cfg.rgb_topic}"
            )
            return False

        if "depth" in self.cfg.data_types and (
            self.depth is None or not self.has_depth_camera_info
        ):
            self.node.get_logger().warn(
                f"Depth data not available yet in {self.cfg.depth_topic}"
            )
            return False

        if "pose" in self.cfg.data_types and self.pose is None:
            self.node.get_logger().warn(
                f"Pose data not available yet in {self.cfg.poses_tf}"
            )
            return False

        self._ready = True
        return True

    def camera_info_callback(self, data: CameraInfo):
        """
        Callback for the camera info
        """
        self.width = data.width
        self.height = data.height
        self.cx = data.k[2]
        self.cy = data.k[5]
        self.fx = data.k[0]
        self.fy = data.k[4]

        self.camera_info = CameraData(
            cx=self.cx,
            cy=self.cy,
            fx=self.fx,
            fy=self.fy,
            width=self.width,
            height=self.height,
        )
        self.has_camera_info = True

    def depth_camera_info_callback(self, data: CameraInfo):
        """
        Callback for the depth camera info
        """
        self.depth_width = data.width
        self.depth_height = data.height
        self.depth_fx = data.k[0]
        self.depth_fy = data.k[4]
        self.depth_cx = data.k[2]
        self.depth_cy = data.k[5]

        self.depth_camera_info = CameraData(
            cx=self.depth_cx,
            cy=self.depth_cy,
            fx=self.depth_fx,
            fy=self.depth_fy,
            width=self.depth_width,
            height=self.depth_height,
        )
        self.has_depth_camera_info = True

    def sync_callback(self, rgb_data: RosImage, depth_data: RosImage):
        """
        Callback for the sync rgb and depth
        """
        self.rgb_callback(rgb_data)
        self.depth_callback(depth_data)

    def rgb_callback(self, data: RosImage):
        """
        Callback for the rgb image
        """
        self.rgb = self.cv_bridge.imgmsg_to_cv2(data, "rgb8")
        self.rgb_timestamp = data.header.stamp
        if "pose" in self.cfg.data_types:
            try:
                geometry_msg_pose = self.tf_buffer.lookup_transform(
                    self.cfg.origin_tf,
                    self.cfg.poses_tf,
                    self.node.get_clock().now(),
                )
                self.pose = (
                    np.array(
                        [
                            geometry_msg_pose.transform.translation.x,
                            geometry_msg_pose.transform.translation.y,
                            geometry_msg_pose.transform.translation.z,
                        ]
                    ),
                    Rotation.from_quat(
                        [
                            geometry_msg_pose.transform.rotation.x,
                            geometry_msg_pose.transform.rotation.y,
                            geometry_msg_pose.transform.rotation.z,
                            geometry_msg_pose.transform.rotation.w,
                        ]
                    ),
                )
            except Exception as e:
                self.node.get_logger().warn(f"Failed to lookup transform: {e}")

    def depth_callback(self, data: RosImage):
        """
        Callback for the depth image
        """
        depth = (
            self.cv_bridge.imgmsg_to_cv2(data, "passthrough").astype(np.float32)
            / 1000.0
        )  # Convert to meters
        self.depth_timestamp = data.header.stamp

        # Count how many pixels are zero
        zero_ratio = np.count_nonzero(depth == 0) / depth.size

        if zero_ratio > 0.7:
            depth[:, :] = 0.0  # Zero out the entire image

        # Apply median filter directly on float32 array
        depth_filtered = median_filter(
            depth, size=7
        )  # You can change size to 3, 7, etc.

        # Resize depth to RGB resolution if available
        if self.has_camera_info and self.has_depth_camera_info:
            depth_resized = cv2.resize(
                depth_filtered,
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )

            # Calculate scaling factors
            scale_x = self.width / self.depth_width
            scale_y = self.height / self.depth_height

            # Adjust intrinsics accordingly
            scaled_fx = self.depth_fx * scale_x
            scaled_fy = self.depth_fy * scale_y
            scaled_cx = self.depth_cx * scale_x
            scaled_cy = self.depth_cy * scale_y

            self.depth_camera_info = CameraData(
                cx=scaled_cx,
                cy=scaled_cy,
                fx=scaled_fx,
                fy=scaled_fy,
                width=self.width,
                height=self.height,
            )
            self.depth = depth_resized
        else:
            self.depth = depth_filtered

    def get_data(self) -> dict:
        """
        Get data from the bridge
        """
        if not self._ready and not self.check_ready():
            return None

        data = {}
        if "rgb" in self.cfg.data_types:
            if self.rgb is not None:
                data["rgb"] = self.rgb.copy()
                self.rgb = None
                data["timestamp"] = self.rgb_timestamp
                self.rgb_timestamp = None
            else:
                # self.node.get_logger().warn("RGB data not available")
                return None

        if "depth" in self.cfg.data_types:
            if self.depth is not None:
                data["depth"] = self.depth.copy()
                self.depth = None
                if "timestamp" not in data.keys():
                    data["timestamp"] = self.depth_timestamp
                    self.depth_timestamp = None

            else:
                # self.node.get_logger().warn("Depth data not available")
                return None

        # If rgb and depth are requested, resize the depth to match the rgb
        if "rgb" in self.cfg.data_types and "depth" in self.cfg.data_types:
            data["depth"] = cv2.resize(data["depth"], (self.width, self.height))

        if "pose" in self.cfg.data_types:
            if self.pose is None:
                # self.node.get_logger().warn("Pose data not available")
                return None
            else:
                data["pose"] = self.pose.copy()
                self.pose = None

        if "semantic" in self.cfg.data_types:
            if self.semantic_gt is not None:
                data["semantic_gt"] = self.semantic_gt
            else:
                data["semantic_gt"] = np.zeros((self.height, self.width)).astype(
                    np.uint8
                )

        # # If any of the data is not available, return None
        # if any([v is None for v in data.values()]):
        #     return None

        return data

    def get_pose(self) -> Tuple[np.ndarray, Rotation]:
        """
        Get the pose from the bridge
        """
        return self.pose
