from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import List, Literal, Tuple

from PIL import Image
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation

from sensors_tools.base.cameras import CameraData

from .base_bridge import BaseBridge, BaseBridgeConfig

ScanNetSensorDataTypes = Literal["rgb", "depth", "semantic", "pose"]
"""
    List of sensor data to query.
    - "rgb": query rgb images.
    - "depth": query depth images.
    - "semantic": query semantic images.
    - "pose": query poses.
"""

@dataclass
class ScanNetBridgeConfig(BaseBridgeConfig):
    """
        Configuration class for ScanNet
    """
    data_types: List[ScanNetSensorDataTypes] = field(default_factory=list, metadata={"default": ["rgb", "poses"]})
    """ Data types to query """

    dataset_path: str = "/root/datasets/scannet"
    """ Path to the dataset """

    scannetv2_labels_combined_path: str = "/root/datasets/scannet/scannetv2-labels.combined.tsv"
    """ Path to the tsv file """

    downsampling_factor_dataset: int = 2
    """ Downsampling factor for the dataset. Or how many images are there in between images. """

    automove: bool = True
    """ Whether to automatically move to the next frame. """

class ScanNetBridge(BaseBridge):
    """
        Bridge for Scannet-like data
    """
    def __init__(self, cfg: ScanNetBridgeConfig):
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
        self.each_n_frame = self.cfg.downsampling_factor_dataset
        self.data_length = len(os.listdir(self.cfg.dataset_path / "color"))
        print("Sequence length: ", self.data_length)
        print("Each n frame: ", self.each_n_frame)
        
        # Sim config data TODO: Move to config file
        #######################################################
        # RELEVANT CAMERA DATA

        # Get camera from the textfile "sequence_name.txt" inside the dataset folder
        # Sequence name is the name of the folder where the data is stored
        self.sequence_name = f"{self.cfg.dataset_path.name}.txt"
        sequence_data_file = open(self.cfg.dataset_path / self.sequence_name, "r")
        sequence_data = sequence_data_file.readlines()
        sequence_data_file.close()

        # Match variables with the textfile
        for line in sequence_data:
            if "colorHeight" in line:
                self.height_color = int(line.split(" ")[2])
            elif "colorWidth" in line:
                self.width_color = int(line.split(" ")[2])
            elif "depthHeight" in line:
                self.height_depth = int(line.split(" ")[2])
            elif "depthWidth" in line:
                self.width_depth = int(line.split(" ")[2])
            elif "fx_color" in line:
                self.fx_color = float(line.split(" ")[2])
            elif "fy_color" in line:
                self.fy_color = float(line.split(" ")[2])
            elif "mx_color" in line:
                self.cx_color = float(line.split(" ")[2])
            elif "my_color" in line:
                self.cy_color = float(line.split(" ")[2])
            elif "fx_depth" in line:
                self.fx_depth = float(line.split(" ")[2])
            elif "fy_depth" in line:
                self.fy_depth = float(line.split(" ")[2])
            elif "mx_depth" in line:
                self.cx_depth = float(line.split(" ")[2])
            elif "my_depth" in line:
                self.cy_depth = float(line.split(" ")[2])
            elif "numColorFrames" in line:
                self.total_steps = int(line.split(" ")[2])
                
        # We adjust to the depth image size
        ratio_width = self.width_depth/self.width_color
        ratio_height = self.height_depth/self.height_color
        self.width_color = self.width_depth
        self.height_color = self.height_depth
        # Adapt cx, cy, fx and fy
        self.cx_color = self.cx_color*ratio_width
        self.cy_color = self.cy_color*ratio_height
        self.fx_color = self.fx_color*ratio_width
        self.fy_color = self.fy_color*ratio_height
        self.camera_info = CameraData(cx=self.cx_color, cy=self.cy_color, fx=self.fx_color, fy=self.fy_color, width=self.width_color, height=self.height_color) 
        print("CAMERA INFO: ", self.camera_info)
        self.depth_camera_info = CameraData(cx=self.cx_depth, cy=self.cy_depth, fx=self.fx_depth, fy=self.fy_depth, width=self.width_depth, height=self.height_depth)
        #######################################################
        label_mapping = pd.read_csv(self.cfg.scannetv2_labels_combined_path, sep='\t')
        NYU_classes = np.array(label_mapping['nyu40id'].astype(int).values)
        eigen_classes = np.array(label_mapping['eigen13id'].astype(int).values)
        scannet_classes = np.array(label_mapping['id'].astype(int).values)
        num_scannet_classes = scannet_classes.max() + 1
        self.remap_to_NYU_classes = - np.ones(num_scannet_classes).astype(int)
        self.remap_to_NYU_classes[0] = 0
        self.remap_to_eigen_classes = - np.ones(num_scannet_classes).astype(int)
        self.remap_to_eigen_classes[0] = 0
        for scannet_id, nyu_id in zip(scannet_classes, NYU_classes):
            self.remap_to_NYU_classes[scannet_id] = int(nyu_id)
        for scannet_id, eigen_id in zip(scannet_classes, eigen_classes):
            self.remap_to_eigen_classes[scannet_id] = int(eigen_id)
        
        # Init pose
        pose_path = self.cfg.dataset_path / "pose" / f"{self.seq_n}.txt"
        translation_matrix = np.loadtxt(pose_path)
        translation = translation_matrix[:3, 3]
        rotation = Rotation.from_matrix(translation_matrix[:3, :3])
        self.pose = (translation, rotation)

        self.ready = True
    
    def open_images(self):
        """
        Open the images from the dataset folder
        Stores the data in a dictionary
        """
        data = {}

        start = time.time()
        if "rgb" in self.cfg.data_types:
            # Load RGB image
            img_path = self.cfg.dataset_path / "color" / f"{self.seq_n}.jpg"
            # Open image as a np array
            img = (Image.open(img_path)).convert('RGB')
            img = img.resize((self.depth_camera_info.width, self.depth_camera_info.height)) #Resize to match the depth image
            # img = img.crop((80, 0, 560, 480)) #Crop the image to match the depth image
            data["rgb"] = np.array(img)
            print(f"Time to load RGB: {time.time() - start}")

        start = time.time()
        if "semantic" in self.cfg.data_types:
            # Load GT label
            label_path = self.cfg.dataset_path / "label" / f"{self.seq_n}.png"
            label = np.array(Image.open(label_path))
            label = self.remap_to_NYU_classes[label]
            print(f"LABEL: {np.unique(label)}")
            label[np.where(label == 255)] = 0 #Remove the white contour
            label = cv2.resize(label, (self.depth_camera_info.width, self.depth_camera_info.height), interpolation = cv2.INTER_NEAREST)
            # label = label[:, 80:560]
            data["semantic_gt"] = label
            print(f"Time to load semantic: {time.time() - start}")

        start = time.time()
        if "depth" in self.cfg.data_types:
            # Load depth image (depth frames as 16-bit pngs (depth shift 1000))
            depth_path = self.cfg.dataset_path / "depth" / f"{self.seq_n}.png"
            depth = np.array(Image.open(depth_path))
            depth = (depth/1000).astype(np.float32)
            # depth = depth[:, 80:560]
            data["depth"] = depth
            print(f"Time to load depth: {time.time() - start}")
        return data

    
    def get_data(self) -> dict:
        """
            Get data from the bridge
        """
        data = {}
        if "pose" in self.cfg.data_types:
            pose_path = self.cfg.dataset_path / "pose" / f"{self.seq_n}.txt"
            translation_matrix = np.loadtxt(pose_path)
            translation = translation_matrix[:3, 3]
            rotation = Rotation.from_matrix(translation_matrix[:3, :3])
            data["pose"] = (translation, rotation)
            self.pose = (translation, rotation)

        img_data = self.open_images()

        data.update(img_data)

        if self.cfg.automove:
            self.move()
            
        return data

    def get_pose(self) -> Tuple[np.ndarray, Rotation]:
        """
            Get pose from the bridge
        """
        return self.pose
    
    def move(self) -> bool:
        """
            Apply increment seq as moving the sensor
        """
        self.seq_n += self.each_n_frame
        if self.seq_n >= self.total_steps:
            return False
        return True