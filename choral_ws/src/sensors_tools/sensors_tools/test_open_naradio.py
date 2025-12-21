#
#
# This file instantiates a sensor and prints the results (mainly for testing semantic inference)
#
#

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from sensor import SensorConfig, SemanticSegmentationSensor
from sensors_tools.bridges.test_bridge import TestBridgeConfig
from sensors_tools.inference.semantic_segmentation.semantic_segmentation_naradio import (
    SemanticSegmentationNaradioConfig,
)

if __name__ == "__main__":
    # Setup the sensor
    bridge_cfg = TestBridgeConfig(
        data_types=["rgb", "semantic", "depth", "pose"],
        dataset_path=Path("./bridges/test_data/dataset/"),
        width=512,
        height=512,
    )
    sem_cfg = SemanticSegmentationNaradioConfig(
        semantic_feature_type="probability_vector",
        semantic_dataset_type="custom_set",
        custom_set_labels=["grass", "person", "sheep", "house", "trees", "dog"],
        colors=[[0, 255, 0], [255, 0, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0]],

    )
    cfg = SensorConfig(bridge_cfg=bridge_cfg, bridge_type="test", inference_type="naradio", inference_cfg=sem_cfg)

    sensor = SemanticSegmentationSensor(cfg)
    sensor.setup()

    fig, ax = plt.subplots(1, 4)

    # Show the images
    for i in range(5):
        data = sensor.get_data()
        if data is None:
            print("Sensor is not ready")
            continue
        ax[0].imshow(data["rgb"])
        # Titles
        ax[0].set_title("RGB")

        print(f"Depth min: {np.min(data['depth'])}, max: {np.max(data['depth'])}")
        ax[1].imshow(data["depth"], cmap="jet")
        ax[1].set_title("Depth")

        ax[2].imshow(data["semantic_gt"])
        ax[2].set_title("Semantic GT")

        ax[3].imshow(data["semantic_rgb"])
        ax[3].set_title("Semantic pred")

        print("Pose: ", data["pose"])

        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
