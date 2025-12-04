from dataclasses import dataclass
import time
from typing import List, Optional, Tuple
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from sensors_tools.inference.models.naradio.naradio import NARadioEncoder
from sensors_tools.inference.models.naradio.utils import compute_cos_sim

from .semantic_segmentation_base import (
    SemanticSegmentationBase,
    SemanticSegmentationBaseConfig,
)
from .semantic_types import SemanticFeatureType
from .semantic_utils import (
    SemanticDatasetType,
    get_labels_color_map,
    get_labels_name,
    labels_to_image,
)


@dataclass
class SemanticSegmentationNaradioConfig(SemanticSegmentationBaseConfig):
    """
    Configuration class for Semantic Segmentation with Naradio
    """

    semantic_dataset_type: SemanticDatasetType = "cityscapes"
    """ Semantic dataset type """

    custom_set_labels: Optional[List[str]] = None
    """ List of custom labels to use """

    semantic_feature_type: SemanticFeatureType = "probability_vector"
    """ Semantic feature type """

    model_version: str = "radio_v2.5-b"
    """ Naradio model version: radio_v2.5-x where x can be b,l, or g """

    lang_model: str = "siglip"
    """ Naradio language model: siglip, clip """

    input_resolution: Tuple[int,int] = (720,720)
    """ Input resolution of the model """

    colors: Optional[List[List[int]]] = None
    """ Optional colormap to use """

class SemanticSegmentationNaradio(SemanticSegmentationBase):
    supported_feature_types = [
        "label",
        "probability_vector",
        "feature_vector",
    ]

    def __init__(
        self,
        cfg: SemanticSegmentationNaradioConfig,
    ):
        self.device = self.init_device(cfg.device)

        self.cfg = cfg

        if self.cfg.semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {self.cfg.semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

        # Config the classes to detect
        if self.cfg.semantic_dataset_type == "custom_set":
            self.label_names = self.cfg.custom_set_labels
        else:
            self.label_names = get_labels_name(self.cfg.semantic_dataset_type)

        model, transform = self.init_model(
            self.cfg.model_version,
            self.cfg.lang_model,
            self.cfg.input_resolution
        )

        # Config the dataset type
        if self.cfg.semantic_dataset_type == "custom_set":
            if self.label_names is None:
                raise ValueError(
                    "custom_set_labels must be provided if semantic_dataset_type is CUSTOM_SET"
                )
            self.semantics_color_map = get_labels_color_map(
                self.cfg.semantic_dataset_type,
                num_classes=len(self.label_names),
            )
            if self.cfg.colors is not None:
                if len(self.cfg.colors) != len(self.label_names):
                    raise ValueError(
                        "colors must have the same length as custom_set_labels"
                    )
                self.semantics_color_map = self.cfg.colors
        else:
            self.semantics_color_map = get_labels_color_map(
                self.cfg.semantic_dataset_type
            )

        # Config label names
        if self.cfg.semantic_dataset_type != "custom_set":
            # TODO: We have to add the prompt engineered ones
            self.label_names = get_labels_name(self.cfg.semantic_dataset_type)

        with torch.no_grad():
            print(f"Input to encode: {self.label_names}")
            self.text_features = model.encode_labels(self.label_names)

        super().__init__(model, transform, self.device, self.cfg.semantic_feature_type)

    def init_model(
        self,
        model_version,
        lang_model,
        input_resolution,
    ):
        model = NARadioEncoder(model_version=model_version, lang_model=lang_model,
            input_resolution=input_resolution, compile=False)

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(model.input_resolution, 
                                  interpolation=transforms.InterpolationMode.BILINEAR)
            ]
        )

        return model, preprocess

    def init_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device(
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
        if device.type == "cuda":
            print("SemanticSegmentationNaradio: Using CUDA")
        elif device.type == "mps":
            if (
                not torch.backends.mps.is_available()
            ):  # Should return True for MPS availability
                raise Exception("SemanticSegmentationNaradio: MPS is not available")
            print("SemanticSegmentationNaradio: Using MPS")
        else:
            print("SemanticSegmentationNaradio: Using CPU")
        return device

    def compute_similarity(self, feat_map):
        # Flatten feat_map_resized keeping its shape to recover later
        N,C = self.text_features.shape
        _, H, W = feat_map.shape
        M = H * W

        feat_map_flat = feat_map.reshape(C, M).T
        sim_matrix_flat = compute_cos_sim(self.text_features, feat_map_flat, softmax=True)
        sim_matrix_flat = sim_matrix_flat.T
        probs = sim_matrix_flat.reshape(N, H, W)

        return probs
    
    @torch.no_grad()
    def infer(self, image):
        prev_width = image.shape[1]
        prev_height = image.shape[0]

        recover_size = torch.nn.Upsample(
            (prev_height, prev_width),
            mode="nearest",
        )
        
        image_pil = Image.fromarray(image)
        img_t = self.transform(image_pil).unsqueeze(0).to(self.device)

        feat_map = self.model.encode_image_to_feat_map(img_t)
        feat_map = self.model.align_spatial_features_with_language(feat_map)
        probs = self.compute_similarity(feat_map[0]).unsqueeze(0)
        # TODO(anonym): Moving to CPU takes 0.8s and it's the biggest bottleneck
        self.labels = recover_size(probs)[0].permute(1, 2, 0).argmax(dim=-1).cpu().numpy() # Save to show if needed

        if self.semantic_feature_type == "feature_vector":
            self.semantics = recover_size(feat_map)[0].permute(1, 2, 0).cpu().numpy()
            return self.semantics

        if self.semantic_feature_type == "probability_vector":
            self.semantics = recover_size(probs).permute(1, 2, 0).cpu().numpy()
            return self.semantics

        # Get the label
        pred = self.labels
        self.semantics = pred

        return self.semantics

    @torch.no_grad()
    def to_rgb(
        self, semantics, bgr=False, feature_type=None, rgb_image=None, overlay=False
    ):

        semantic_feature_type = (
            feature_type if feature_type is not None else self.semantic_feature_type
        )

        if semantic_feature_type == "label":
            return labels_to_image(
                semantics,
                self.semantics_color_map,
                bgr=bgr,
                overlay=overlay,
                rgb_image=rgb_image,
            )
        elif semantic_feature_type == "probability_vector":
            return labels_to_image(
                np.argmax(semantics, axis=-1),
                self.semantics_color_map,
                bgr=bgr,
                overlay=overlay,
                rgb_image=rgb_image,
            )
        elif semantic_feature_type == "feature_vector":
            return labels_to_image(
                self.labels, # Do this trick to speed up things atm TODO
                self.semantics_color_map,
                bgr=bgr,
                overlay=overlay,
                rgb_image=rgb_image,
            )

    def get_semantic_dimensions(self):
        if self.semantic_feature_type == "label":
            return 1
        elif self.semantic_feature_type == "probability_vector":
            return self.semantics_color_map.shape[0]
        elif self.semantic_feature_type == "feature_vector":
            return self.text_features.shape[1] # Dimensions
