#
#
# This file instantiates a sensor and prints the results (mainly for testing semantic inference)
#
#

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from sensor import SensorConfig, SemanticSegmentationSensor
from sensors_tools.bridges.folder_bridge import FolderBridgeConfig
from sensors_tools.inference.semantic_segmentation.semantic_segmentation_trident import (
    SemanticSegmentationTridentConfig,
)
import torch

if __name__ == "__main__":
    # Setup the sensor
    bridge_cfg = FolderBridgeConfig(
        data_types=["rgb", "semantic"],
        dataset_path=Path("/root/datasets/folder/"),
    )
    sem_cfg = SemanticSegmentationTridentConfig(
        semantic_feature_type="probability_vector",
        semantic_dataset_type="custom_set",
        custom_set_labels=[
            "rubble",
            "road",
            "grass",
            "house",
            "tree",
            "water",
            "car",
            "sand",
        ],
        colors=[
            [0, 0, 0],
            [125, 125, 125],
            [100, 255, 100],
            [217, 82, 255],
            [0, 150, 0],
            [0, 255, 255],
            [255, 0, 0],
            [255, 255, 0],
        ],
    )
    cfg = SensorConfig(
        bridge_cfg=bridge_cfg,
        bridge_type="folder",
        inference_type="trident",
        inference_cfg=sem_cfg,
        save_inference=True,
        overlay=True,
        save_inference_path=Path("/root/datasets/folder/out/"),
    )
    # Show if cuda is available
    print(f"Cuda available: {torch.cuda.is_available()}")
    sensor = SemanticSegmentationSensor(cfg)
    sensor.setup()

    # Process dataset
    sensor.get_data()
    while sensor.move():
        sensor.get_data()
