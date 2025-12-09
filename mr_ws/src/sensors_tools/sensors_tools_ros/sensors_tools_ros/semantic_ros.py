from dataclasses import fields
import dataclasses
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros import TransformBroadcaster

from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
from std_srvs.srv import SetBool

from poses_tools.frame_converter import FrameConverter

from sensors_tools.sensor import SemanticSegmentationSensor, SensorConfig
from sensors_tools.bridges import ControllableBridges, BridgeType, get_bridge_config
from sensors_tools.inference.semantic_segmentation.semantic_types import (
    SemanticSegmentationMethods,
)
from sensors_tools.inference.semantic_segmentation import (
    get_semantic_segmentation_config,
)

from sensors_tools_msgs.srv import MoveSensor


class SemanticNode(Node):
    def __init__(self):
        super().__init__("semantic_node")
        self.setup()

    def setup(self):
        self.get_logger().info("Setting up semantic node")
        # Get params
        self.load_config()

        # Asserts are placed here for typing purposes
        assert isinstance(self.frequency, float), "Frequency must be a float"
        assert isinstance(self.camera_name, str), "Camera name must be a string"

        self.sensor = SemanticSegmentationSensor(self.cfg)
        self.sensor.setup()

        self.controllable_bridge = (
            self.sensor.cfg.bridge_type in ControllableBridges
        )  # enables the move service OR move with timer

        self.pose_transformer = FrameConverter()
        self.pose_transformer.setup_transform_function(
            self.sensor.cfg.bridge_type, "ros"
        )

        #######################################################
        # RELEVANT CAMERA DATA is checked at callbacks due to executors
        self.camera_info = None
        self.depth_camera_info = None
        #######################################################

        self.data_types = self.sensor.cfg.bridge_cfg.data_types
        # ROS setup depending on the queried sensors
        if "pose" in self.data_types:
            self.tf_broadcaster = TransformBroadcaster(self)
            self.pub_camera_odometry = self.create_publisher(
                Odometry, f"/{self.camera_name}/odom", qos_profile_sensor_data
            )
            # self.pose_timer = self.create_timer(0.001, self.update_and_publish_odom)

        if "rgb" in self.data_types:
            self.pub_rgb = self.create_publisher(
                ImageMsg, f"/{self.camera_name}/rgb/image_raw", qos_profile_sensor_data
            )

        if "depth" in self.data_types:
            self.pub_depth = self.create_publisher(
                ImageMsg,
                f"/{self.camera_name}/depth/image_raw",
                qos_profile_sensor_data,
            )
            self.pub_point_cloud = self.create_publisher(
                PointCloud2, f"/{self.camera_name}/point_cloud", qos_profile_sensor_data
            )
            if self.publish_freespace_point_cloud:
                self.pub_freespace_point_cloud = self.create_publisher(
                    PointCloud2,
                    f"/{self.camera_name}/freespace_point_cloud",
                    qos_profile_sensor_data,
                )
        if "semantic" in self.data_types:
            assert (
                self.cfg.inference_cfg is not None
            ), "Inference cfg must be specified if semantic data is requested"
            self.pub_semantic_gt = self.create_publisher(
                ImageMsg,
                f"/{self.camera_name}/semantic_gt/image_raw",
                qos_profile_sensor_data,
            )
            self.pub_semantic = self.create_publisher(
                ImageMsg,
                f"/{self.camera_name}/semantic/image_raw",
                qos_profile_sensor_data,
            )

        if self.frequency == -1:
            self.srv_capture_data = self.create_service(
                SetBool, f"/{self.camera_name}/capture_data", self.loop_srv
            )
            if self.controllable_bridge:
                self.srv_move = self.create_service(
                    SetBool, f"/{self.camera_name}/move", self.move_srv
                )
                self.srv_move_to_pose = self.create_service(
                    MoveSensor,
                    f"/{self.camera_name}/move_to_pose",
                    self.move_to_pose_srv,
                )
        else:
            interval = int(1.0 / self.frequency)
            self.timer = self.create_timer(interval, self.loop)

        self.get_logger().info("Finished semantic node setup")

    def load_config(self):
        """
        Load config from rosparams
        """
        # Parameter declaration
        bridge_param = self.declare_parameter(
            "bridge.bridge_type", "ros"
        )  # Can't be uninitialized due to https://github.com/ros2/rclpy/issues/1460

        bridge_type: BridgeType = bridge_param.get_parameter_value().string_value
        print(f"Bridge type: {bridge_type}")
        bridge_config_class = get_bridge_config(bridge_type)

        bridge_parameters = self.load_rosparams(bridge_config_class, "bridge")
        bridge_cfg = bridge_config_class(**bridge_parameters)

        # We have to manually add this node to the config if bridge type is ros
        if bridge_type == "ros":
            bridge_cfg.node = self

        semantic_segmentation_type_param = self.declare_parameter(
            "semantic_segmentation_type", "segformer"
        )
        semantic_segmentation_type: SemanticSegmentationMethods = (
            semantic_segmentation_type_param.get_parameter_value().string_value
        )

        print(f"Semantic segmentation type: {semantic_segmentation_type}")

        inference_config_class = get_semantic_segmentation_config(
            semantic_segmentation_type
        )
        inference_parameters = self.load_rosparams(inference_config_class, "inference")
        inference_cfg = inference_config_class(**inference_parameters)

        # TODO: This should be managed by the bridge
        # gt_label_mapper_param = self.declare_parameter("gt_labels_mapper", None)
        # gt_label_mapper = gt_label_mapper_param.value

        save_inference_param = self.declare_parameter("save_inference", False)
        save_inference: bool = save_inference_param.get_parameter_value().bool_value
        if save_inference:
            save_inference_path_param = self.declare_parameter(
                "save_inference_path", None
            )
            save_inference_path = Path(
                save_inference_path_param.get_parameter_value().string_value
            )
        else:
            save_inference_path = None

        self.cfg = SensorConfig(
            bridge_type=bridge_type,
            bridge_cfg=bridge_cfg,
            inference_cfg=inference_cfg,
            inference_type=semantic_segmentation_type,
            save_inference=save_inference,
            save_inference_path=save_inference_path,
        )

        print(f"Loaded Sensor")

        if "depth" in self.cfg.bridge_cfg.data_types:
            stride_param = self.declare_parameter("stride", 1)
            self.stride = stride_param.get_parameter_value().integer_value

            max_range_param = self.declare_parameter("max_range", -1.0)
            self.max_range = max_range_param.get_parameter_value().double_value

            publish_freespace_point_cloud_param = self.declare_parameter(
                "publish_freespace_point_cloud", False
            )
            self.publish_freespace_point_cloud = (
                publish_freespace_point_cloud_param.get_parameter_value().bool_value
            )

        frequency_param = self.declare_parameter("frequency", 10.0)
        self.frequency = frequency_param.get_parameter_value().double_value
        camera_name_param = self.declare_parameter("camera_name", "camera_sensor")
        self.camera_name = camera_name_param.get_parameter_value().string_value

        # The sensor will provide a transformation that will depend on the parameters.
        # The bridge will do its own thing.
        # IMPORTANT:
        # - The sensor_frame_id will be included in the pcl message!
        # - Any other static transform should be defined in URDF or the launch file
        #   Typical examples of this are: odom -> map, and camera_link -> base_link
        # - Be careful with duplicated sources of TF.
        #   (e.g. if you have a SLAM system, you don't need the sensor to provide pose)
        origin_frame_param = self.declare_parameter("origin_frame_id", "odom")
        self.origin_frame_id = origin_frame_param.get_parameter_value().string_value
        sensor_frame_param = self.declare_parameter("sensor_frame_id", "camera_link")
        self.sensor_frame_id = sensor_frame_param.get_parameter_value().string_value

    def load_rosparams(self, config_class, namespace: str = "") -> dict:
        """
        Load rosparams from the bridge config template
        and use it to initialize the bridge config class
        """
        parameters = {}

        # Get bridge config class fields
        config_fields = fields(config_class)
        # Get bridge config rosparams
        for field in config_fields:
            field_name = field.name

            param_name = f"{namespace}.{field_name}" if namespace else field_name

            if isinstance(field.default, dataclasses._MISSING_TYPE):
                print(f"Getting default value withOUT type: {type(field.metadata)}")
                default_value = field.metadata.get("default")
            else:
                print(f"Getting default value with type: {type(field.default)}")
                default_value = field.default

            param = self.declare_parameter(param_name, default_value)
            param_value = param.value
            print(f"Param value: {param_value}")

            # Convert to Path if field name contains 'path'
            if "path" in field_name and param_value is not None:
                param_value = Path(param_value)

            parameters[field_name] = param_value

            print(f"Loaded: {field_name} = {param_value}")

        return parameters

    def to_odom_msg(
        self, translation: np.ndarray, rotation: Rotation, timestamp: Time
    ) -> Odometry:
        """
        Convert a 4x4 pose matrix to a ROS Odometry message
        """
        odom_msg = Odometry()
        odom_msg.header.frame_id = self.origin_frame_id
        odom_msg.header.stamp = timestamp
        odom_msg.child_frame_id = self.sensor_frame_id
        odom_msg.pose.pose.position.x = translation[0]
        odom_msg.pose.pose.position.y = translation[1]
        odom_msg.pose.pose.position.z = translation[2]
        quat = rotation.as_quat(canonical=False)
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]
        return odom_msg

    def odom_msg_to_tf_msg(self, odom_msg: Odometry) -> TransformStamped:
        """
        Convert a ROS Odometry message to a ROS TransformStamped message
        """
        tf_msg = TransformStamped()
        print(f"odom msg with timestamp: {odom_msg.header.stamp}")
        tf_msg.header.stamp = odom_msg.header.stamp
        tf_msg.header.frame_id = odom_msg.header.frame_id
        tf_msg.child_frame_id = odom_msg.child_frame_id
        tf_msg.transform.translation.x = odom_msg.pose.pose.position.x
        tf_msg.transform.translation.y = odom_msg.pose.pose.position.y
        tf_msg.transform.translation.z = odom_msg.pose.pose.position.z
        tf_msg.transform.rotation.x = odom_msg.pose.pose.orientation.x
        tf_msg.transform.rotation.y = odom_msg.pose.pose.orientation.y
        tf_msg.transform.rotation.z = odom_msg.pose.pose.orientation.z
        tf_msg.transform.rotation.w = odom_msg.pose.pose.orientation.w
        return tf_msg

    def update_and_publish_odom(self):
        # Update the odometry
        translation, rotation = self.sensor.bridge.get_pose()
        translation_ros, rotation_ros = self.pose_transformer.transform_function(
            translation, rotation
        )
        odom_msg = self.to_odom_msg(
            translation_ros, rotation_ros, self.get_clock().now().to_msg()
        )  # TODO(anonym): check if we want to pass the time directly instead of msg
        self.tf_broadcaster.sendTransform(self.odom_msg_to_tf_msg(odom_msg))
        self.pub_camera_odometry.publish(odom_msg)

    def pcd_from_rgb_depth(
        self, rgb: np.ndarray, depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """A function that converts an rgb image and a depth image to a point cloud
        Depth image is measured with respect to image plane.
        Depth is along X axis

        Args:
            rgb: PIL Image
            depth: PIL Image

        Returns:
            pcd: a point cloud as a numpy array with shape (H*W, 4)
        """
        rgb_np = rgb
        depth_np = depth
        H, W = depth_np.shape

        columns, rows = np.meshgrid(
            np.linspace(0, W - 1, num=int(W / self.stride)),
            np.linspace(0, H - 1, num=int(H / self.stride)),
        )  # type: ignore
        point_depth = depth_np[:: self.stride, :: self.stride]
        depth_mask = (
            (point_depth > 0) & (point_depth < self.max_range)
            if self.max_range != -1.0
            else (point_depth > 0)
        )

        y = (
            -(columns - self.depth_camera_info.cx)
            * point_depth
            / (self.depth_camera_info.fx)
        )  # Originally x : Now -y
        z = (
            -(rows - self.depth_camera_info.cy)
            * point_depth
            / (self.depth_camera_info.fy)
        )  # Originally y : Now -z
        x = point_depth  # Originally z : Now x
        pcd = np.dstack((x, y, z)).astype(np.float32)

        pcd[~depth_mask] = 0

        colors = rgb_np[:: self.stride, :: self.stride, :]
        colors = colors.astype(np.uint8)
        # We have to create a new np.array to swap channels and use the view function later
        # Add alpha channel
        colors = np.dstack(
            (
                colors[:, :, 2],
                colors[:, :, 1],
                colors[:, :, 0],
                np.ones((colors.shape[0], colors.shape[1], 1), dtype=np.uint8) * 255,
            )
        )

        return pcd, colors

    def generate_freespace_point_cloud_msg(
        self, depth: np.ndarray, time_stamp: Time
    ) -> PointCloud2:
        """
        Find the freespace points in the point cloud that are further than
        max_range and set them to that depth sothe volumetric map can use
        the freespace information
        """
        depth_np = depth

        if self.max_range == -1.0:
            # Return an empty point cloud
            return PointCloud2()

        # Threshold the depth image
        depth_np[depth_np > self.max_range] = self.max_range
        W = depth_np.shape[1]
        H = depth_np.shape[0]
        columns, rows = np.meshgrid(
            np.linspace(0, W - 1, num=int(W / self.stride)),
            np.linspace(0, H - 1, num=int(H / self.stride)),
        )  # type: ignore
        point_depth = depth_np[:: self.stride, :: self.stride]
        y = (
            -(columns - self.depth_camera_info.cx)
            * point_depth
            / (self.depth_camera_info.fx)
        )  # Originally x : Now -y
        z = (
            -(rows - self.depth_camera_info.cy)
            * point_depth
            / (self.depth_camera_info.fy)
        )  # Originally y : Now -z
        x = point_depth  # Originally z : Now x
        pcd = np.dstack((x, y, z)).astype(np.float32)
        # Only keep the points are max_range
        mask = depth_np[:: self.stride, :: self.stride] == self.max_range
        pcd = pcd[mask]

        # Create a point cloud message
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header.stamp = time_stamp
        point_cloud_msg.header.frame_id = self.sensor_frame_id
        point_cloud_msg.height = 1
        point_cloud_msg.width = pcd.shape[0]

        point_cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud_msg.is_bigendian = False
        point_cloud_msg.point_step = 12
        point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
        point_cloud_msg.is_dense = True
        point_cloud_msg.data = pcd.tobytes()

        return point_cloud_msg

    def generate_point_cloud_msg(
        self, points_pcd: np.ndarray, points_RGB: np.ndarray, time_stamp: Time
    ) -> PointCloud2:
        """
        Generate a point cloud message

        Args:
            points_pcd: a numpy array with shape (H*W, 3)
            points_RGB: a numpy array with shape (H*W, 4)
            time_stamp: a rospy.Time object
        """
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header.stamp = time_stamp
        point_cloud_msg.header.frame_id = self.sensor_frame_id
        point_cloud_msg.height = 1

        point_cloud_msg.width = points_pcd.shape[0] * points_pcd.shape[1]
        point_cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT8, count=1),
        ]

        point_cloud_msg.is_bigendian = False
        point_cloud_msg.point_step = 16
        point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
        point_cloud_msg.is_dense = True

        colors_converted = points_RGB.view(np.float32)
        points_data = np.dstack(
            (
                points_pcd,
                colors_converted,
            )
        )
        point_cloud_msg.data = points_data.tobytes()
        return point_cloud_msg

    def generate_point_cloud_semantics_msg(
        self,
        points_pcd: np.ndarray,
        points_RGB: np.ndarray,
        semantic: np.ndarray,
        semantic_gt: np.ndarray,
        time_stamp: Time,
    ):
        """
        Generate a point cloud message with semantics using the pcd from the RGB-D images

        Args:
            points_pcd: a numpy array with shape (H*W, 3)
            points_RGB: a numpy array with shape (H*W, 4)
            semantic: a numpy array with shape (H, W, C) with C the number of classes
            semantic_gt: a numpy array with shape (H, W) with the ground truth class
            time_stamp: a rospy.Time object
        """
        # Generate the point cloud message from a point cloud of size
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header.stamp = time_stamp
        point_cloud_msg.header.frame_id = self.sensor_frame_id
        point_cloud_msg.height = 1
        # print(f"Semantic values: {semantic}")
        point_cloud_msg.width = points_pcd.shape[0] * points_pcd.shape[1]

        sent_n_classes = self.sensor.inference_model.get_semantic_dimensions()  # type: ignore

        # The PointField is defined with a name, the starting byte offset, the data type and number of elements.
        # rgb is encoded in the standard way so RViz can visualize it. It will be transformed to float32 with view
        # gt_semantics includes a single value with the ground truth class. It is of type float because it is easier to modify in the dstack
        # semantics includes the probabilities of the classes. It has as many elements as classes
        point_cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            PointField(
                name="gt_semantics", offset=16, datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="semantics",
                offset=20,
                datatype=PointField.FLOAT32,
                count=sent_n_classes,
            ),
        ]
        point_cloud_msg.is_bigendian = False
        point_cloud_msg.point_step = 20 + 4 * sent_n_classes
        point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
        point_cloud_msg.is_dense = True

        # Turn numpy matrix points_RGB of 128 128 4 of type uint8 into a numpy array of size 128 128 1 of type float32
        colors_converted = points_RGB.view(np.float32)
        gt_class_converted = semantic_gt.astype(np.float32)
        points_data = np.dstack(
            (
                points_pcd,
                colors_converted,
                gt_class_converted[:: self.stride, :: self.stride],
                semantic[:: self.stride, :: self.stride, :],
            )
        )
        point_cloud_msg.data = points_data.tobytes()

        return point_cloud_msg

    def check_camera_info(self):
        if self.camera_info and self.depth_camera_info:
            return True

        if self.cfg.bridge_type != "ros":
            self.camera_info = (
                self.sensor.bridge.camera_info if "rgb" in self.data_types else None
            )
            self.depth_camera_info = (
                self.sensor.bridge.depth_camera_info
                if "depth" in self.data_types
                else None
            )
        else:
            if "rgb" in self.data_types:
                if self.sensor.bridge.has_camera_info:
                    self.camera_info = self.sensor.bridge.camera_info
                else:
                    self.get_logger().info(
                        f"ROS bridge waiting for camera info on: {self.sensor.bridge.cfg.camera_info_topic}"
                    )
                    return False
            if "depth" in self.data_types:
                if self.sensor.bridge.has_depth_camera_info:
                    self.depth_camera_info = self.sensor.bridge.depth_camera_info
                else:
                    self.get_logger().info(
                        f"ROS bridge waiting for depth camera info on: {self.sensor.bridge.cfg.camera_info_topic}"
                    )
                    return False

        return True

    def loop(self):
        """
        Loop that captures data and publishes it
        """
        self.check_camera_info()

        # Get data TODO: The ready or not ready should be in the sensor itself with the bridge and inference info
        data = self.sensor.get_data()
        if data is None:
            # self.get_logger().warn("Sensor is not ready")
            return
        else:
            self.get_logger().info("Got data!")

        if self.sensor.cfg.bridge_type == "ros":
            timestamp = data["timestamp"]
        else:
            timestamp = self.get_clock().now().to_msg()

        # Publish data
        if "pose" in self.data_types:
            print("Publishing pose.")
            translation, rotation = data["pose"]
            translation_ros, rotation_ros = self.pose_transformer.transform_function(
                translation, rotation
            )
            odom_msg = self.to_odom_msg(translation_ros, rotation_ros, timestamp)
            tf_msg = self.odom_msg_to_tf_msg(odom_msg)
            self.tf_broadcaster.sendTransform(tf_msg)
            self.pub_camera_odometry.publish(odom_msg)

        if "rgb" in self.data_types:
            try:
                rgb_msg = CvBridge().cv2_to_imgmsg(data["rgb"], "rgb8")
                if self.pub_rgb.get_subscription_count() > 0:
                  print("Publishing rgb.")
                  self.pub_rgb.publish(rgb_msg)
            except CvBridgeError as e:
                print(e)

        if "semantic" in self.data_types:
            if "semantic_gt" in data and data["semantic_gt"] is not None:
                try:
                    semantic_gt_img = self.sensor.inference_model.to_rgb(
                        data["semantic_gt"], feature_type="label"
                    )
                    semantic_gt_msg = CvBridge().cv2_to_imgmsg(semantic_gt_img, "rgb8")
                    self.pub_semantic_gt.publish(semantic_gt_msg)
                except CvBridgeError as e:
                    print(e)

            semantic_msg = CvBridge().cv2_to_imgmsg(data["semantic_rgb"], "rgb8")
            if self.pub_semantic.get_subscription_count() > 0:
              print("Publishing 2D semantics.")
              self.pub_semantic.publish(semantic_msg)

        if "depth" in self.data_types:
            try:
                depth_msg = CvBridge().cv2_to_imgmsg(data["depth"], "passthrough")
                if self.pub_depth.get_subscription_count() > 0:
                  print("Publishing depth.")
                  self.pub_depth.publish(depth_msg)
            except Exception as e:
                print(e)
            # Publish point cloud
            points_pcd, points_RGB = self.pcd_from_rgb_depth(data["rgb"], data["depth"])
            
            if self.publish_freespace_point_cloud and self.pub_freespace_point_cloud.get_subscription_count() > 0:
                free_pcd_msg = self.generate_freespace_point_cloud_msg(
                    data["depth"], timestamp
                )
                
                self.pub_freespace_point_cloud.publish(free_pcd_msg)

            if "semantic" in self.data_types:
                assert (
                    self.sensor.cfg.inference_cfg is not None
                ), "Inference not configured!"

                pcd_msg = self.generate_point_cloud_semantics_msg(
                    points_pcd,
                    points_RGB,
                    data["semantic"],
                    data["semantic_gt"],
                    timestamp,
                )
                if self.pub_point_cloud.get_subscription_count() > 0:
                  print("Publishing point cloud with semantics.")
                  self.pub_point_cloud.publish(pcd_msg)
            else:
                pcd_msg = self.generate_point_cloud_msg(
                    points_pcd, points_RGB, timestamp
                )
                self.pub_point_cloud.publish(pcd_msg)
                # print("Send point cloud with: ", pcd_msg.width, " points")

    def move_to_pose_srv(self, req: MoveSensor.Request) -> MoveSensor.Response:
        """
        Move the camera
        """
        translation = req.pose.pose.position
        translation = np.array([translation.x, translation.y, translation.z])
        quat = req.pose.pose.orientation
        rotation = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])

        self.sensor.bridge.move_to_pose(translation, rotation)

        response = MoveSensor.Response()
        response.success = True
        return response

    def move_srv(self, req: SetBool.Request) -> SetBool.Response:
        """
        Move the camera
        """
        success = self.sensor.bridge.move()

        response = SetBool.Response()
        response.success = success
        return response

    def loop_srv(self, req: SetBool.Request) -> SetBool.Response:
        """
        Loop that captures data and publishes it
        """
        self.loop(None)

        response = SetBool.Response()
        response.success = True
        return response
