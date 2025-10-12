from dataclasses import dataclass
import numpy as np

import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_MobileNet_V3_Large_Weights,
)
from torchvision import transforms

from .semantic_segmentation_base import SemanticSegmentationBase, SemanticSegmentationBaseConfig
from .semantic_types import SemanticFeatureType
from .semantic_utils import SemanticDatasetType, get_labels_color_map, labels_to_image

@dataclass
class SemanticSegmentationDeepLabV3Config(SemanticSegmentationBaseConfig):
  """
      Configuration class for Semantic Segmentation with DeepLabV3
  """
  
  encoder_name: str = "resnet50"
  """ Encoder name for DeepLabV3 """
  
  semantic_dataset_type: SemanticDatasetType = "voc"
  """ Semantic dataset type """

  semantic_feature_type: SemanticFeatureType = "probability_vector"
  """ Semantic feature type """

  model_path: str = ""
  """ Path to load a pre-trained model """


class SemanticSegmentationDeepLabV3(SemanticSegmentationBase):
    model_configs = {
        "resnet50": {
            "encoder": "resnet50",
            "model": deeplabv3_resnet50,
            "weights": DeepLabV3_ResNet50_Weights.DEFAULT,
            "dataset": "voc",
            "image_size": (512, 512),
        },
        "resnet101": {
            "encoder": "resnet101",
            "model": deeplabv3_resnet101,
            "weights": DeepLabV3_ResNet101_Weights.DEFAULT,
            "dataset": "voc",
            "image_size": (512, 512),
        },
        "mobilenetv3": {
            "encoder": "mobilenetv3",
            "model": deeplabv3_mobilenet_v3_large,
            "weights": DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
            "dataset": "voc",
            "image_size": (512, 512),
        },
    }
    supported_feature_types = [
        "label",
        "probability_vector",
    ]

    def __init__(
        self,
        cfg: SemanticSegmentationDeepLabV3Config,
    ):
        device = self.init_device(cfg.device)

        self.cfg = cfg

        model, transform = self.init_model(
            device, self.cfg.encoder_name, self.cfg.model_path, self.cfg.semantic_dataset_type
        )

        self.semantics_color_map = get_labels_color_map(self.cfg.semantic_dataset_type)

        self.semantic_dataset_type = self.cfg.semantic_dataset_type

        if self.cfg.semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {self.cfg.semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

        super().__init__(model, transform, device, self.cfg.semantic_feature_type)

    def init_model(self, device, encoder_name, model_path):
        if encoder_name not in self.model_configs:
            raise ValueError(
                f"Encoder name {encoder_name} is not supported for {self.__class__.__name__}"
            )
        model = self.model_configs[encoder_name]["model"](
            self.model_configs[encoder_name]["weights"]
        )
        if model_path != "":  # Load pre-trained models
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.to(device).eval()
        transform = self.model_configs[encoder_name]["weights"].transforms()
        return model, transform

    def init_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device(
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
        if device.type == "cuda":
            print("SemanticSegmentationDeepLabV3: Using CUDA")
        elif device.type == "mps":
            if (
                not torch.backends.mps.is_available()
            ):  # Should return True for MPS availability
                raise Exception("SemanticSegmentationDeepLabV3: MPS is not available")
            print("SemanticSegmentationDeepLabV3: Using MPS")
        else:
            print("SemanticSegmentationDeepLabV3: Using CPU")
        return device

    @torch.no_grad()
    def infer(self, image):
        prev_width = image.shape[1]
        prev_height = image.shape[0]
        recover_size = transforms.Resize(
            (prev_height, prev_width),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        image_torch = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        batch = self.transform(image_torch).unsqueeze(0)
        prediction = self.model(batch)["out"]
        probs = prediction.softmax(dim=1)
        probs = recover_size(probs[0])

        if self.semantic_feature_type == "label":
            self.semantics = probs.argmax(dim=0).cpu().numpy()
        elif self.semantic_feature_type == "probability_vector":
            self.semantics = probs.permute(1, 2, 0).cpu().numpy()

        return self.semantics

    def to_rgb(self, semantics, bgr=False, feature_type=None, overlay=False,rgb_image=None):
        
        semantic_feature_type = feature_type if feature_type is not None else self.semantic_feature_type

        if semantic_feature_type == "label":
            return labels_to_image(semantics, self.semantics_color_map, bgr=bgr, overlay=overlay, rgb_image=rgb_image)
        elif semantic_feature_type == "probability_vector":
            return labels_to_image(
                np.argmax(semantics, axis=-1), self.semantics_color_map, bgr=bgr, overlay=overlay, rgb_image=rgb_image)

    def get_semantic_dimensions(self):
        if self.semantic_feature_type == "label":
            return 1
        elif self.semantic_feature_type == "probability_vector":
            return self.semantics_color_map.shape[0]
        