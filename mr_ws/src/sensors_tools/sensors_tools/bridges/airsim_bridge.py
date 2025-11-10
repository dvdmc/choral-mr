from dataclasses import dataclass, field
import time
from typing import List, Literal, Tuple
from threading import Lock

from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation

import airsim  # type: ignore

from airsim_tools.depth_conversion import depth_conversion  # type: ignore
from airsim_tools.semantics import get_airsim_labels, rgb2label  # type: ignore

from sensors_tools.base.cameras import CameraData
from sensors_tools.bridges.base_bridge import BaseBridge, BaseBridgeConfig
from tqdm import tqdm

AirsimSensorDataTypes = Literal["rgb", "depth", "semantic", "pose"]
"""
    List of sensor data to query.
    - "pose": query poses.
    - "rgb": query rgb images.
    - "depth": query depth images.
    - "semantic": query semantic images.
"""


@dataclass
class AirsimBridgeConfig(BaseBridgeConfig):
    """
    Configuration class for AirsimBridge
    """

    data_types: List[AirsimSensorDataTypes] = field(
        default_factory=list, metadata={"default": ["rgb", "pose"]}
    )
    """ Data types to query """

    semantic_config: List[List] = field(default_factory=list, metadata={"default": []})
    """ Semantic configuration for Airsim. It's a list with name-id pairs"""

    width: int = 512
    """ Image width """

    height: int = 512
    """ Image height """

    fov_h: float = 54.4
    """ Horizontal field of view """


class AirsimBridge(BaseBridge):
    """
    Bridge for Airsim
    """

    def __init__(self, cfg: AirsimBridgeConfig):
        """
        Constructor
        """
        super().__init__(cfg)
        self.cfg = cfg
        self.ready = False

    def setup(self):
        """
        Setup the bridge
        """
        # Data acquisition configuration
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()
        # Clien mutex
        self.client_mutex = Lock()

        # Sim config data TODO: Move to config file
        #######################################################
        # RELEVANT CAMERA DATA
        self.width = self.cfg.width
        self.height = self.cfg.height
        self.fov_h = self.cfg.fov_h
        self.cx = float(self.width) / 2
        self.cy = float(self.height) / 2
        fov_h_rad = self.fov_h * np.pi / 180.0
        self.fx = self.cx / (np.tan(fov_h_rad / 2))
        self.fy = self.fx * self.height / self.width
        self.client.simSetFocusAperture(7.0, "0")  # Avoids depth of field blur
        self.client.simSetFocusDistance(100.0, "0")  # Avoids depth of field blur
        self.camera_info = CameraData(
            cx=self.cx,
            cy=self.cy,
            fx=self.fx,
            fy=self.fy,
            width=self.width,
            height=self.height,
        )
        self.depth_camera_info = CameraData(
            cx=self.cx,
            cy=self.cy,
            fx=self.fx,
            fy=self.fy,
            width=self.width,
            height=self.height,
        )
        #######################################################

        # Set the data to query
        self.query_data = []
        if "rgb" in self.cfg.data_types:
            self.query_data.append(
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            )
        if "depth" in self.cfg.data_types:
            self.query_data.append(
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
            )
        if "semantic" in self.cfg.data_types:
            self.query_data.append(
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
            )
        if "semantic" in self.cfg.data_types:
            self.setup_semantic_config()

        self.ready = True

    def setup_semantic_config(self):
        # Set all objects in the scene to label 0 in the beggining
        if self.cfg.semantic_config is not None:
            # Set everything to ID 0 using regular expression
            success = self.client.simSetSegmentationObjectID(".*", 0, True)

            # To change the remaining we use the semantic config.
            # For each label, we will create a regular expression
            # that matches all the objects containing the label as a substring
            regexes = {}
            for label, label_id in self.cfg.semantic_config:
                if label not in regexes:
                    regexes[label] = ".*" + label + ".*"
                else:
                    regexes[label] += "|.*" + label + ".*"
            print("Setting object IDs")
            for label, label_id in tqdm(self.cfg.semantic_config):
                success = self.client.simSetSegmentationObjectID(
                    regexes[label], label_id, True
                )

            print("Finished setting object IDs")

    def process_img_responses(self, responses: List[airsim.ImageResponse]) -> dict:
        """
        Process the data from AirSim.
        """
        img_data = {}

        for response in responses:
            if response.image_type == airsim.ImageType.Scene:
                np_image = np.frombuffer(
                    response.image_data_uint8, dtype=np.uint8
                ).reshape(response.height, response.width, 3)
                correct_image = np_image[:, :, ::-1]

                img_data["rgb"] = correct_image.copy()
            elif response.image_type == airsim.ImageType.DepthPerspective:
                img_depth_meters = airsim.list_to_2d_float_array(
                    response.image_data_float, response.width, response.height
                )
                img_depth_meters_corrected = depth_conversion(img_depth_meters, self.fx)
                img_data["depth"] = np.array(img_depth_meters_corrected).astype(
                    np.float32
                )

            elif response.image_type == airsim.ImageType.Segmentation:
                # Transform Airsim segmentation image to a different color system
                img_buffer = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb_airsim = img_buffer.reshape(response.height, response.width, 3)
                np_rgb_airsim = img_rgb_airsim[:, :, ::-1]
                # Get the semantic image
                semantic = rgb2label(np_rgb_airsim, get_airsim_labels())
                img_data["semantic_gt"] = np.array(semantic)

        return img_data

    def get_data(self) -> dict:
        """
        Get data from the bridge
        """
        data = {}
        if "pose" in self.cfg.data_types:
            # Acquire the client mutex
            self.client_mutex.acquire()
            pose = self.client.simGetVehiclePose()
            self.client_mutex.release()
            translation = np.array(
                [pose.position.x_val, pose.position.y_val, pose.position.z_val]
            )
            quat = np.array(
                [
                    pose.orientation.x_val,
                    pose.orientation.y_val,
                    pose.orientation.z_val,
                    pose.orientation.w_val,
                ]
            )
            rotation = Rotation.from_quat(quat)
            data["pose"] = (translation, rotation)
        self.client_mutex.acquire()
        start = time.time()
        responses = self.client.simGetImages(self.query_data)
        print(f"Time to get images: {time.time() - start}")
        self.client_mutex.release()
        start = time.time()
        img_data = self.process_img_responses(responses)
        print(f"Time to process images: {time.time() - start}")
        data.update(img_data)

        return data

    def get_pose(self) -> Tuple[np.ndarray, Rotation]:
        """
        Get pose from the bridge
        """
        self.client_mutex.acquire()
        pose = self.client.simGetVehiclePose()
        self.client_mutex.release()
        translation = np.array(
            [pose.position.x_val, pose.position.y_val, pose.position.z_val]
        )
        quat = np.array(
            [
                pose.orientation.x_val,
                pose.orientation.y_val,
                pose.orientation.z_val,
                pose.orientation.w_val,
            ]
        )
        rotation = Rotation.from_quat(quat)
        return (translation, rotation)
