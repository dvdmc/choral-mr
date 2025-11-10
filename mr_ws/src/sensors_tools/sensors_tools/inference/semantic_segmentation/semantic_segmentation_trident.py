from dataclasses import dataclass
from typing import List, Optional
import numpy as np

import torch
from PIL import Image
from torchvision import transforms
from sensors_tools.inference.models.trident.trident import Trident


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
class SemanticSegmentationTridentConfig(SemanticSegmentationBaseConfig):
    """
    Configuration class for Semantic Segmentation with Trident
    """

    semantic_dataset_type: SemanticDatasetType = "cityscapes"
    """ Semantic dataset type """

    custom_set_labels: Optional[List[str]] = None
    """ List of custom labels to use """

    semantic_feature_type: SemanticFeatureType = "probability_vector"
    """ Semantic feature type """

    sam_checkpoint_path: str = "/root/checkpoints/Trident/sam_vit_b_01ec64.pth"
    """ Path to the SAM checkpoint """

    sam_model_type: str = "vit_b"
    """ Type of the SAM model """

    coarse_threshold: float = 0.2
    """ Threshold for the SAM refinement """

    colors: Optional[List[List[int]]] = None
    """ Optional colormap to use """


class SemanticSegmentationTrident(SemanticSegmentationBase):
    supported_feature_types = [
        "label",
        "probability_vector",
    ]

    def __init__(
        self,
        cfg: SemanticSegmentationTridentConfig,
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
            self.label_names,
            self.cfg.coarse_threshold,
            self.cfg.sam_checkpoint_path,
            self.cfg.sam_model_type,
        )
        self.softmax = torch.nn.Softmax(dim=0).to(self.device)

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

        super().__init__(model, transform, self.device, self.cfg.semantic_feature_type)

    def init_model(
        self,
        label_names,
        coarse_threshold,
        sam_checkpoint_path,
        sam_model_type,
    ):
        model = Trident(
            clip_type="openai",
            clip_model_type="ViT-B/16",
            vfm_model="dino",
            class_names=label_names,
            device=self.device,
            sam_refinement=True,
            coarse_thresh=coarse_threshold,
            minimal_area=225,
            debug=False,
            sam_ckpt=sam_checkpoint_path,
            sam_model_type=sam_model_type,
        )

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model = model.eval()

        return model, preprocess

    def init_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device(
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
        if device.type == "cuda":
            print("SemanticSegmentationTrident: Using CUDA")
        elif device.type == "mps":
            if (
                not torch.backends.mps.is_available()
            ):  # Should return True for MPS availability
                raise Exception("SemanticSegmentationTrident: MPS is not available")
            print("SemanticSegmentationTrident: Using MPS")
        else:
            print("SemanticSegmentationTrident: Using CPU")
        return device

    @torch.no_grad()
    def infer(self, image):
        prev_width = image.shape[1]
        prev_height = image.shape[0]
        recover_size = transforms.Resize(
            (prev_height, prev_width),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        image_pil = Image.fromarray(image)
        img_t = self.transform(image_pil).unsqueeze(0).to(self.device)

        pred, logits = self.model.predict(image_pil, img_t)
        probs = self.softmax(logits)
        probs = recover_size(probs)

        if self.semantic_feature_type == "probability_vector":
            self.semantics = probs.permute(1, 2, 0).cpu().numpy()
            return self.semantics

        # Get the label
        pred = probs.argmax(dim=0)
        self.semantics = pred.cpu().numpy()

        return self.semantics

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
            print(f"LABELS: {np.unique(np.argmax(semantics, axis=-1))}")
            return labels_to_image(
                np.argmax(semantics, axis=-1),
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
