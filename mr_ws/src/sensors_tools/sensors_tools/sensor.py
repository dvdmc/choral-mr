from dataclasses import dataclass
from pathlib import Path
import time
import random  # Just for setting the random seed
from typing import Optional

import numpy as np
from PIL import Image

from sensors_tools.bridges import BridgeConfig, BridgeType, get_bridge

from sensors_tools.inference.semantic_segmentation import InferenceConfig
from sensors_tools.inference.semantic_segmentation import get_semantic_segmentation
from sensors_tools.utils.random_utils import set_seed

from sensors_tools.inference.semantic_segmentation.semantic_types import (
    SemanticSegmentationMethods,
)


@dataclass
class SensorConfig:
    """
    Configuration class for classicSensor
    """

    bridge_cfg: BridgeConfig
    """ Bridge configuration """

    bridge_type: BridgeType
    """ Type of bridge to be used """

    # NOTE: The following is responsability of the bridge. Will be moved in the future
    # gt_labels_mapper: Optional[str] = None
    # """ Name reference the label map for derived datasets. Example: coco_voc_2_pascal_8 """

    inference_cfg: Optional[InferenceConfig] = None
    """ Inference configuration """

    inference_type: SemanticSegmentationMethods = None
    """ Type of inference to be used """

    save_inference: bool = False
    """ Whether to save the inference results """

    overlay: bool = False
    """ Whether to overlay the inference results on the rgb image """

    save_inference_path: Optional[Path] = None
    """ Path to save the inference results """


class SemanticSegmentationSensor:
    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg
        self.seq = 0
        set_seed(42)

    def setup(self):
        # Setup the inference models.
        # This is done before the bridge to allow for loading the model and weights before starting the robot.
        print("Setting up the inference model")
        if "semantic" in self.cfg.bridge_cfg.data_types:
            assert (
                self.cfg.inference_cfg is not None
            ), "Inference cfg must be specified if semantic data is requested"
            # Dump inference_cfg
            print(self.cfg.inference_cfg)
            self.inference_model = get_semantic_segmentation(
                self.cfg.inference_type, self.cfg.inference_cfg
            )

            if self.cfg.save_inference:
                print("Saving inference results")
                assert (
                    self.cfg.save_inference_path is not None
                ), "save_inference_path must be specified if save_inference is True"
                self.pred_path = self.cfg.save_inference_path / "pred"
                self.pred_rgb_path = self.cfg.save_inference_path / "pred_rgb"
                self.pred_path.mkdir(parents=True, exist_ok=True)
                self.pred_rgb_path.mkdir(parents=True, exist_ok=True)
                print(f"Results will be saved in {self.cfg.save_inference_path}")
                print(f"Pred rgb will be saved in {self.pred_rgb_path}")

        # Setup the bridge
        print("Setting up the bridge")
        self.bridge = get_bridge(self.cfg.bridge_type, self.cfg.bridge_cfg)
        self.bridge.setup()

        # If there is a GT label mapper, load it. NOTE: This is now handled by the bridge TODO(anonym)
        # print(self.cfg.gt_labels_mapper)
        # self.gt_labels_mapper = None
        # if self.cfg.gt_labels_mapper is not None:
        #     print("Found labels mapper")
        #     self.gt_labels_mapper = get_label_mapper(self.cfg.gt_labels_mapper)

    def get_data(self) -> Optional[dict]:
        if not self.bridge.ready:
            return None

        data = self.bridge.get_data()
        if data is None:
            return None


        img = data["rgb"]
        if "semantic" in self.cfg.bridge_cfg.data_types:
            assert (
                self.cfg.inference_cfg is not None
            ), "Inference cfg must be specified if semantic data is requested"

            start = time.time()
            semantics = self.inference_model.infer(img)
            print(f"Time to get prediction: {time.time() - start}")

            # NOTE: This is now handled by the bridge
            # if self.gt_labels_mapper is not None and "semantic_gt" in data:
            #     data["semantic_gt"] = apply_label_map(data["semantic_gt"], self.gt_labels_mapper)

            data["semantic"] = semantics
            data["semantic_rgb"] = self.inference_model.to_rgb(
                semantics, overlay=self.cfg.overlay, rgb_image=img
            )

            if self.cfg.save_inference:
                print(f"Saving: {self.pred_rgb_path}/{self.seq}.png")
                np.save(self.pred_path / f"{self.seq}.npy", data["semantic"])
                semantic_rgb = Image.fromarray(data["semantic_rgb"])
                semantic_rgb.save(self.pred_rgb_path / f"{self.seq}.png")
                self.seq += 1

        return data

    def move(self) -> bool:
        return self.bridge.move()
