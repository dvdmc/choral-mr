from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from sensors_tools.inference.open_clip_semantic import OpenClipSemanticSegmentation, OpenClipSemanticSegmentationConfig
from sensor import SensorConfig, SemanticSegmentationSensor
from sensors_tools.bridges.test_bridge import TestBridgeConfig

if __name__ == '__main__':
    # Setup the sensor configuration
    bridge_cfg = TestBridgeConfig(
        data_types=["rgb", "semantic", "depth", "pose"],
        dataset_path=Path("./bridges/test_data/dataset/"),
        width=512,
        height=512,
    )
    sem_cfg = OpenClipSemanticSegmentationConfig(
        model_name="clip", # TODO: Implement support for different encoders
        num_classes=2,
        classes_text="human person woman",
        skip_center_crop=True,
        labels_name="binary",
    )
    cfg = SensorConfig(
        bridge_cfg=bridge_cfg,
        bridge_type="test",
        inference_cfg=sem_cfg,
    )

    # Initialize the sensor
    sensor = SemanticSegmentationSensor(cfg)
    sensor.setup()

    fig, ax = plt.subplots(1, 4)

    # Process and visualize data
    for i in range(5):
        data = sensor.get_data()
        if(data is None):
            print("Sensor is not ready")
        else:    
            # Display RGB image
            ax[0].imshow(data["rgb"])
            ax[0].set_title("RGB")

            # Display depth map
            print(f"Depth min: {np.min(data['depth'])}, max: {np.max(data['depth'])}")
            ax[1].imshow(data["depth"], cmap="jet")
            ax[1].set_title("Depth")

            # Display semantic ground truth
            ax[2].imshow(data["semantic_gt"])
            ax[2].set_title("Semantic GT")

            # Perform prediction and display semantic overlay
            open_inference = OpenClipSemanticSegmentation(sem_cfg)
            open_inference.setup()
            pred_data = open_inference.get_prediction(data["rgb"])
            ax[3].imshow(pred_data["img_out"])
            ax[3].set_title("Semantic pred")

            print("Pose: ", data["pose"])

        # Update visualization
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
