###
#
# You can use this file in combination with the test_data and generator.py to test certain loading functionalities.
#
###
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import List, Literal, Tuple

from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation

from sensors_tools.base.cameras import CameraData
from sensors_tools.bridges.base_bridge import BaseBridge, BaseBridgeConfig

TestSensorDataTypes = Literal["rgb", "depth", "semantic", "pose"]
"""
    List of sensor data to query.
    - "pose": query pose.
    - "rgb": query rgb images.
    - "depth": query depth images.
    - "semantic": query semantic images.
"""

@dataclass
class TestBridgeConfig(BaseBridgeConfig):
    """
        Configuration class for Test
    """
    dataset_path: Path = Path("./test_data/")
    """ Path to the dataset """
    
    width: int = 512
    """ Image width """

    height: int = 512
    """ Image height """


class TestBridge(BaseBridge):
    """
        Bridge for Airsim
    """
    def __init__(self, cfg: TestBridgeConfig):
        """
            Constructor
        """
        super().__init__(cfg)
        self.cfg = cfg

    
    def setup(self):
        """
            Setup the bridge
        """
        # Data acquisition configuration
        self.static_tf = []
        print("Dataset path: ", self.cfg.dataset_path)
        self.seq_n = 0
        self.data_length = len(os.listdir(self.cfg.dataset_path / "color"))
        print("Sequence length: ", self.data_length)
        
        # Sim config data TODO: Move to config file
        #######################################################
        # RELEVANT CAMERA DATA

        # Get camera from the json "camera_intrinsics.json" inside the dataset folder
        with open(self.cfg.dataset_path / "camera_intrinsics.json", "r") as f:
            camera_data = json.load(f)
        
        print("Camera data: ", camera_data)
        self.camera_info = CameraData(cx=camera_data["cx"], cy=camera_data["cy"], fx=camera_data["fx"], fy=camera_data["fy"], width=self.cfg.width, height=self.cfg.height)
        #######################################################
    
        self.ready = True

    def open_images(self):
        """
        Open the images from the dataset folder
        Stores the data in a dictionary
        """
        data = {}

        if "rgb" in self.cfg.data_types:
            # Load RGB image
            img_path = self.cfg.dataset_path / "color" / f"{self.seq_n:04d}.png"
            # Open image as a np array
            img = (Image.open(img_path)).convert('RGB')
            data["rgb"] = np.array(img)
        
        if "semantic" in self.cfg.data_types:
            # Load GT label
            label_path = self.cfg.dataset_path / "label" / f"{self.seq_n:04d}.png"
            label = Image.open(label_path)
            data["semantic_gt"] = np.array(label)

        if "depth" in self.cfg.data_types:
            # Load depth image (depth frames as 16-bit pngs (depth shift 1000))
            depth_path = self.cfg.dataset_path / "depth" / f"{self.seq_n:04d}.png"
            depth = np.array(Image.open(depth_path))
            depth = (depth/1000).astype(np.float32)
            data["depth"] = depth

        return data

    
    def get_data(self) -> dict:
        """
            Get data from the bridge
        """
        data = {}
        if "pose" in self.cfg.data_types:
            pose_path = self.cfg.dataset_path / "pose" / f"{self.seq_n:04d}.txt"
            pose_matrix = np.loadtxt(pose_path)
            translation = pose_matrix[:3, 3]
            rotation = Rotation.from_matrix(pose_matrix[:3, :3])
            data["pose"] = (translation, rotation)
            self.pose = (translation, rotation)
        img_data = self.open_images()

        data.update(img_data)

        # Assume seq advances (TODO: Figure out what to do in general for datasets)
        self.increment_seq()

        return data
    

    def get_pose(self) -> Tuple[np.ndarray, Rotation]:
        """
            Get pose from the bridge
        """
        return self.pose
    
    def move(self):
        """
            Apply increment seq as moving the sensor
        """
        self.increment_seq()
        return True
    
    def increment_seq(self):
        """
            Increment the sequence number
        """
        self.seq_n += 1
        if self.seq_n == self.data_length:
            self.seq_n = 0
        # print("Sequence number: ", self.seq_n)