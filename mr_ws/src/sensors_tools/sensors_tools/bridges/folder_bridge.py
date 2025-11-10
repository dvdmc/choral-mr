###
#
# You can use this file in combination with the test_data and generator.py to test certain loading functionalities.
#
###
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation

from sensors_tools.base.cameras import CameraData
from sensors_tools.bridges.base_bridge import BaseBridge, BaseBridgeConfig

FolderSensorDataTypes = Literal["rgb"]


@dataclass
class FolderBridgeConfig(BaseBridgeConfig):
    """
    Configuration class for FolderBridge
    """

    dataset_path: Path = Path("/root/datasets/folder")
    """ Path to the dataset """

    data_types: List[FolderSensorDataTypes] = field(
        default_factory=list, metadata={"default": ["rgb"]}
    )
    width: int = 2000
    height: int = 1500
    """ Data types to query """


class FolderBridge(BaseBridge):
    """
    Bridge for folder data
    """

    def __init__(self, cfg: FolderBridgeConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def setup(self):
        """
        Setup the bridge
        """
        # Data acquisition configuration
        print("Dataset path: ", self.cfg.dataset_path)
        self.files = os.listdir(self.cfg.dataset_path)
        self.files.sort()
        self.data_length = len(self.files)
        self.seq_n = 0
        print("Sequence length: ", self.data_length)

        self.ready = True

    def get_data(self) -> dict:
        """
        Get data from the bridge
        """
        data = {}
        if "rgb" in self.cfg.data_types:
            if self.seq_n < len(self.files):
                # Load RGB image
                img_path = self.cfg.dataset_path / self.files[self.seq_n]
                # Open image as a np array
                img = (Image.open(img_path)).convert("RGB")
                img = img.resize((self.cfg.width, self.cfg.height))
                data["rgb"] = np.array(img)
            else:
                print("No more files to load")

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
