from dataclasses import dataclass
from typing import List, Optional
import cv2
from einops import rearrange
import numpy as np

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose
from f3rm.features.clip import clip as f3rm_clip
from f3rm.features.clip import tokenize

from .semantic_segmentation_base import SemanticSegmentationBase, SemanticSegmentationBaseConfig
from .semantic_types import SemanticFeatureType
from .semantic_utils import (
    SemanticDatasetType,
    similarity_heatmap_image,
    get_labels_color_map,
    get_labels_name,
    labels_to_image,
)

@dataclass
class SemanticSegmentationCLIPConfig(SemanticSegmentationBaseConfig):
    """
        Configuration class for Semantic Segmentation with CLIP
    """
    encoder_name: str = "ViT-L/14@336px"
    """ Name of the CLIP model to use """

    semantic_dataset_type: SemanticDatasetType = "cityscapes"
    """ Semantic dataset type """

    semantic_feature_type: SemanticFeatureType = "probability_vector"
    """ Semantic feature type """
    
    model_path: Optional[str] = None
    """ Path to the CLIP model to use """

    sim_text_query: str = "clock"
    """ Text query to use for similarity heatmap """
    
    skip_center_crop: bool = True
    """ Skip center crop """

    custom_set_labels: Optional[List[str]] = None
    """ List of custom labels to use """

    colors: Optional[List[List[int]]] = None
    """ List of custom colors to use """

class SemanticSegmentationCLIP(SemanticSegmentationBase):
    # CLIP available models: https://github.com/f3rm/f3rm/tree/main/f3rm/features/clip
    available_configs = [
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        "ViT-B/32",
        "ViT-B/16",
        "ViT-L/14",
        "ViT-L/14@336px",
    ]

    supported_feature_types = [
        "label",
        "probability_vector",
        "feature_vector",
    ]

    # TODO(dvdmc): take the text query parameter out
    def __init__(
        self,
        cfg: SemanticSegmentationCLIPConfig,
    ):
        device = self.init_device(cfg.device)

        self.cfg = cfg

        # NOTE: transform is called preprocess in the original code
        model, transform = self.init_model(
            device, self.cfg.encoder_name
        )

        if self.cfg.semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {self.cfg.semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

        # Config the dataset type
        if self.cfg.semantic_dataset_type == "custom_set":
            if self.cfg.custom_set_labels is None:
                raise ValueError(
                    "custom_set_labels must be provided if semantic_dataset_type is CUSTOM_SET"
                )
            self.semantics_color_map = get_labels_color_map(
                self.cfg.semantic_dataset_type, num_classes=len(self.cfg.custom_set_labels)
            )
            if self.cfg.colors is not None:
              if len(self.cfg.colors) != len(self.cfg.custom_set_labels):
                raise ValueError(
                    "colors must have the same length as custom_set_labels if semantic_dataset_type is CUSTOM_SET"
                )
              self.semantics_color_map = self.cfg.colors
        elif self.cfg.semantic_dataset_type == "feature_similarity":
            if self.cfg.semantic_feature_type != "feature_vector":
                raise ValueError(
                    "semantic_feature_type must be FEATURE_VECTOR if semantic_dataset_type is FEATURE_SIMILARITY"
                )
            if self.cfg.sim_text_query == "":
                raise ValueError(
                    "sim_text_query must be provided if semantic_dataset_type is FEATURE_SIMILARITY"
                )
            self.semantics_color_map = None
            self.sim_scale = 3.0  # NOTE: This is for visualization
        else:
            self.semantics_color_map = get_labels_color_map(self.cfg.semantic_dataset_type)
            
        # Config the text encodings
        if self.cfg.semantic_dataset_type == "custom_set":
            self.label_names = self.cfg.custom_set_labels
            self.tokens = [
                tokenize(text_query).to(device) for text_query in self.cfg.custom_set_labels
            ]
        elif self.cfg.semantic_dataset_type == "feature_similarity":
            # We already checked that semantic_feature_type is "feature_vector","
            self.label_names = [self.cfg.sim_text_query]
            self.tokens = [
                tokenize(self.cfg.sim_text_query).to(device)
            ]  # We will only work with a single text query
        else:
            self.label_names = get_labels_name(self.cfg.semantic_dataset_type)
            self.tokens = torch.stack(
                [tokenize(text_query).to(device) for text_query in self.label_names]
            )  # Shape: (N, token_dim)

        self.text_embs = torch.stack(
            [model.encode_text(token).squeeze(0) for token in self.tokens]
        )

        self.text_embs /= self.text_embs.norm(dim=-1, keepdim=True)

        # Patch the preprocess if we want to skip center crop
        if self.cfg.skip_center_crop:
            # Check there is exactly one center crop transform
            is_center_crop = [isinstance(t, CenterCrop) for t in transform.transforms]
            assert sum(is_center_crop) == 1, (
                "There should be exactly one CenterCrop transform"
            )
            # Create new transform without center crop
            transform = Compose(
                [t for t in transform.transforms if not isinstance(t, CenterCrop)]
            )
            print("Skipping center crop")

        super().__init__(model, transform, device, self.cfg.semantic_feature_type)

    def init_model(
        self, device, encoder_name
    ):
        # Check if selected config is available
        if encoder_name not in self.available_configs:
            raise ValueError(
                f"SemanticSegmentationCLIP does not support {encoder_name} model."
            )

        model, preprocess = f3rm_clip.load(encoder_name, device)
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
            print("SemanticSegmentationCLIP: Using CUDA")
        elif device.type == "mps":
            if (
                not torch.backends.mps.is_available()
            ):  # Should return True for MPS availability
                raise Exception("SemanticSegmentationCLIP: MPS is not available")
            print("SemanticSegmentationCLIP: Using MPS")
        else:
            print("SemanticSegmentationCLIP: Using CPU")
        return device

    def set_query_word(self, query_word):
        if self.cfg.semantic_dataset_type == "feature_similarity":
            self.label_names = [query_word]
            self.tokens = [tokenize(query_word).to(self.device)]
            self.text_embs = torch.stack(
                [self.model.encode_text(token).squeeze(0) for token in self.tokens]
            )
            self.text_embs /= self.text_embs.norm(dim=-1, keepdim=True)
        else:
            print(
                "Setting the query word will have no effect since semantic_dataset_type is not FEATURE_SIMILARITY"
            )

    def get_output_dims(self, h_in, w_in):
        """Compute output dimensions."""
        # from https://github.com/f3rm/f3rm/blob/main/f3rm/features/clip_extract.py
        if self.cfg.encoder_name.startswith("ViT"):
            h_out = h_in // self.model.visual.patch_size
            w_out = w_in // self.model.visual.patch_size
            return h_out, w_out

        if self.cfg.encoder_name.startswith("RN"):
            h_out = max(h_in / w_in, 1.0) * self.model.visual.attnpool.spacial_dim
            w_out = max(w_in / h_in, 1.0) * self.model.visual.attnpool.spacial_dim
            return int(h_out), int(w_out)

        raise ValueError(f"unknown clip model: {self.cfg.encoder_name}")

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

        # We use a batch inference approach for using the implementation in f3rm
        embeddings = []
        embeddings.append(self.model.get_patch_encodings(img_t))

        embeddings = torch.cat(embeddings, dim=0)

        h_out, w_out = self.get_output_dims(img_t.shape[-2], img_t.shape[-1])

        embeddings = rearrange(
            embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out
        )  # (H, W, D)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.squeeze(0)

        if self.semantic_feature_type == "feature_vector":
            # Nothing to do except recover the size
            # We need to permute channels to do the recover size correctly TODO(dvdmc): can we improve this?
            self.semantics = (
                recover_size(embeddings.permute(2, 0, 1)).permute(1, 2, 0).cpu().numpy()
            )
            return self.semantics

        # Compute similarities
        sims = embeddings @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)

        # NOTE: Careful with channel ordering here. It differs from other sem seg modules!

        # Normalize to get "probabilities"
        probs = (sims / sims.norm(dim=-1, keepdim=True)).permute(2, 0, 1)
        probs = recover_size(probs)

        if self.semantic_feature_type == "probability_vector":
            self.semantics = probs.permute(1, 2, 0).cpu().numpy()
            return self.semantics

        # Get the label
        pred = probs.argmax(dim=0)
        # if self.semantic_feature_type == "label": # NOT NECESSARY FOR NOW
        self.semantics = pred.cpu().numpy()
        return self.semantics

    def to_rgb(self, semantics, bgr=False, feature_type=None, overlay=False, rgb_image=None):
        
        semantic_feature_type = feature_type if feature_type is not None else self.semantic_feature_type

        if semantic_feature_type == "label":
            return labels_to_image(semantics, self.semantics_color_map, bgr=bgr, overlay=overlay, rgb_image=rgb_image)
        elif semantic_feature_type == "probability_vector":
            return labels_to_image(
                np.argmax(semantics, axis=-1), self.semantics_color_map, bgr=bgr, overlay=overlay, rgb_image=rgb_image)
            
        elif semantic_feature_type == "feature_vector":
            # Transform semantic to tensor
            # TODO(dvdmc): check if doing these operations (and functions below) in CPU is more efficient (it probably is)
            semantics = torch.from_numpy(semantics).to(self.device)
            # Compute similarity
            sims = semantics @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
            if self.cfg.semantic_dataset_type == "feature_similarity":
                return similarity_heatmap_image(
                    sims.cpu().detach().numpy(),
                    colormap=cv2.COLORMAP_JET,
                    sim_scale=self.sim_scale,
                    bgr=bgr,
                    overlay=overlay,
                    rgb_image=rgb_image,
                )
            else:
                pred = sims.argmax(dim=-1)
                return labels_to_image(
                    pred.cpu().detach().numpy(), self.semantics_color_map, bgr=bgr,
                    overlay=overlay, rgb_image=rgb_image
                )

    def features_to_sims(self, semantics):
        """Public interface to compute similarity

        Args:
            semantics (np.ndarray): Semantic features of generic shape ([1] or [H, W], D)
        """

        if self.semantic_feature_type != "feature_vector":
            print(
                "WARNING: if you computed semantics from this module, they shouldn't be used with features_to_sims()"
            )
        # Transform semantic to tensor
        semantics = torch.from_numpy(semantics).to(self.device)
        # Compute similarity
        sims = semantics @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
        return sims.cpu().detach().numpy()

    def features_to_labels(self, semantics):
        """Public interface to compute labels

        Args:
            semantics (np.ndarray): Semantic features of generic shape ([1] or [H, W], D)
        """
        if self.semantic_feature_type != "feature_vector":
            print(
                "WARNING: if you computed semantics from this module, they shouldn't be used with features_to_labels()"
            )
        # Transform semantic to tensor
        semantics = torch.from_numpy(semantics).to(self.device)
        # Compute similarity
        sims = semantics @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
        pred = sims.argmax(dim=-1)
        return pred.cpu().detach().numpy()

    def get_semantic_dimensions(self):
        if self.semantic_feature_type == "label":
            return 1
        elif self.semantic_feature_type == "probability_vector":
            return self.semantics_color_map.shape[0]
        elif self.semantic_feature_type == "feature_vector":
            return self.text_embs.shape[0]