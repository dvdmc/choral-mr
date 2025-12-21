from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
from torchvision import transforms

from .semantic_labels import get_ade20k_to_scannet40_map
from .semantic_segmentation_base import (
    SemanticSegmentationBase,
    SemanticSegmentationBaseConfig,
)
from .semantic_types import SemanticFeatureType
from .semantic_utils import (
    SemanticDatasetType,
    get_labels_color_map,
    labels_to_image,
)


@dataclass
class SemanticSegmentationSegformerConfig(SemanticSegmentationBaseConfig):
    """
    Configuration class for Semantic Segmentation with Segformer
    """

    model_variant: str = "b0"
    """ Model variant: b0, b1, b2, b3, b4, b5 """

    image_size: tuple = (512, 512)
    """ Size of the input image """

    semantic_dataset_type: SemanticDatasetType = "cityscapes"
    """ Dataset name: cityscapes, ade20k, voc, nyu40, custom_set """

    semantic_feature_type: SemanticFeatureType = "probability_vector"
    """ Semantic feature type: label, probability_vector, feature_vector """

    model_path: Optional[str] = None
    """ Path to the model weights """

    # label_mapping = None Not implemented yet


class SemanticSegmentationSegformer(SemanticSegmentationBase):
    # Segformer available models: https://huggingface.co/models?search=nvidia/segformer
    # They can be configured by:
    # - Model variant: b0, b1, b2, b3, b4, b5
    # - Sizes: (512,512), (512,1024), (768,768), (1024,1024), (640, 1280)
    # - Dataset: cityscapes, ade
    # Check the specific available configurations
    available_configs = [
        ("b0", (1024, 1024), "cityscapes"),
        ("b0", (512, 512), "ade20k"),
        ("b0", (512, 1024), "cityscapes"),
        ("b0", (640, 1280), "cityscapes"),
        ("b0", (768, 768), "cityscapes"),
        ("b1", (1024, 1024), "cityscapes"),
        ("b1", (512, 512), "ade20k"),
        ("b2", (1024, 1024), "cityscapes"),
        ("b2", (512, 512), "ade20k"),
        ("b3", (1024, 1024), "cityscapes"),
        ("b3", (512, 512), "ade20k"),
        ("b4", (1024, 1024), "cityscapes"),
        ("b4", (512, 512), "ade20k"),
        ("b5", (1024, 1024), "cityscapes"),
    ]
    # TODO(anonym): this can be used to make mappings more generic NOTE: not currently used
    available_mappings = [
        {"in": "ade20k", "out": "nyu40", "map": get_ade20k_to_scannet40_map()},
    ]
    supported_feature_types = ["label", "probability_vector"]

    def __init__(
        self,
        cfg: SemanticSegmentationSegformerConfig,
    ):
        self.label_mapping = None
        self.cfg = cfg

        device = self.init_device(cfg.device)

        model, transform = self.init_model(
            device,
            self.cfg.model_variant,
            self.cfg.semantic_dataset_type,
            self.cfg.image_size,
            self.cfg.model_path,
        )

        print(f"Config: {self.cfg}")

        self.semantics_color_map = get_labels_color_map(self.cfg.semantic_dataset_type)

        self.semantic_dataset_type = self.cfg.semantic_dataset_type

        if self.cfg.semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {self.cfg.semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

        super().__init__(model, transform, device, self.cfg.semantic_feature_type)

    def init_model(
        self, device, model_variant, semantic_dataset_type, image_size, model_path
    ):
        # Convert image_size to appropiate form
        if semantic_dataset_type == "cityscapes":
            image_size = (512, 1024)
        else:
            image_size = (512, 512)

        friendly_dataset_type = semantic_dataset_type
        # We allow to use this model on NYU40 by mapping labels from ADE20K
        if semantic_dataset_type == "nyu40":
            self.label_mapping = get_ade20k_to_scannet40_map()
            friendly_dataset_type = "ade20k"

        # Check if selected config is available
        if (
            model_variant,
            image_size,
            friendly_dataset_type,
        ) not in self.available_configs:
            raise ValueError(
                f"Segformer does not support {model_variant} model with size {image_size} and dataset {semantic_dataset_type}"
            )

        # Convert dataset type to appropiate form
        dataset = friendly_dataset_type.lower()
        if dataset == "ade20k":
            dataset = "ade"

        if model_path is None:  # Load pre-trained models
            model = SegformerForSemanticSegmentation.from_pretrained(
                f"nvidia/segformer-{model_variant}-finetuned-{dataset}-{image_size[0]}-{image_size[1]}"
            )
        else:
            raise NotImplementedError(
                f"Segformer only supports pre-trained model for now. You tried to load a custom model: {model_path}"
            )  # TODO(anonym): allow to load a custom model
        model = model.to(device).eval()
        transform = AutoImageProcessor.from_pretrained(
            f"nvidia/segformer-{model_variant}-finetuned-{dataset}-{image_size[0]}-{image_size[1]}"
        )
        return model, transform

    def init_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device(
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
        if device.type == "cuda":
            print("SemanticSegmentationSegformer: Using CUDA")
        elif device.type == "mps":
            if (
                not torch.backends.mps.is_available()
            ):  # Should return True for MPS availability
                raise Exception("SemanticSegmentationSegformer: MPS is not available")
            print("SemanticSegmentationSegformer: Using MPS")
        else:
            print("SemanticSegmentationSegformer: Using CPU")
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
        batch = self.transform(images=image_pil, return_tensors="pt").to(self.device)
        prediction = self.model(**batch).logits
        probs = prediction.softmax(dim=1)
        probs = recover_size(probs[0])

        if self.semantic_feature_type == "label":
            self.semantics = probs.argmax(dim=0).cpu().numpy()
            if self.label_mapping is not None:
                self.semantics = self.label_mapping[self.semantics]

        elif self.semantic_feature_type == "probability_vector":
            self.semantics = probs.permute(1, 2, 0).cpu().numpy()
            if self.label_mapping is not None:
                self.semantics = self.aggregate_probabilities(
                    self.semantics, self.label_mapping
                )

        return self.semantics

    def aggregate_probabilities(
        self, semantics: np.ndarray, label_mapping: np.ndarray
    ) -> np.ndarray:
        """
        Aggregates original probabilities into output probabilities using a label mapping.

        Args:
            semantics: np.ndarray of shape [H, W, original num classes] - softmaxed class probabilities.
            label_mapping: np.ndarray of shape [original num classes] - maps original class indices to output classes.

        Returns:
            np.ndarray of shape [H, W, num output classes] - aggregated probabilities.
        """
        H, W, num_original_classes = semantics.shape
        num_output_classes = label_mapping.max() + 1

        aggregated = np.zeros((H, W, num_output_classes), dtype=semantics.dtype)

        for in_idx, out_idx in enumerate(label_mapping):
            aggregated[..., out_idx] += semantics[..., in_idx]

        return aggregated

    def to_rgb(
        self, semantics, bgr=False, feature_type=None, overlay=False, rgb_image=None
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

    def get_semantic_dimensions(self):
        if self.semantic_feature_type == "label":
            return 1
        elif self.semantic_feature_type == "probability_vector":
            return self.semantics_color_map.shape[0]
