import enum
from typing import Literal
import numpy as np
import cv2

from .semantic_types import SemanticDatasetType

from .semantic_labels import (
    get_ade20k_color_map,
    get_cityscapes_color_map,
    get_coco_color_map,
    get_nyu40_color_map,
    get_voc_color_map,
    get_generic_color_map,
)
from .semantic_labels import (
    get_ade20k_labels,
    get_cityscapes_labels,
    get_nyu40_labels,
    get_voc_labels,
)


def similarity_heatmap_image(
    sim_map,
    colormap=cv2.COLORMAP_JET,
    sim_scale=1.0,
    bgr=False,
    overlay=False,
    alpha=0.5,
    rgb_image=None,
):
    """
    Transforms a similarity map to a visual RGB image using a colormap.

    Args:
        sim_map (np.ndarray): Similarity image of shape (H, W)
        colormap (int): OpenCV colormap (e.g., cv2.COLORMAP_JET)

    Returns:
        np.ndarray: RGB image (H, W, 3) visualizing similarity
    """
    # Normalize to 0–255 for colormap (skip min-max if values are already in [0,1])
    sim_map = np.clip(sim_map * sim_scale, 0.0, 1.0)
    sim_map = ((1 - sim_map) * 255).astype(np.uint8)

    # Apply colormap and convert to RGB
    sim_color = cv2.applyColorMap(sim_map, colormap)
    if bgr:
        sim_color = cv2.cvtColor(sim_color, cv2.COLOR_BGR2RGB)

    # Overlay if requested
    if overlay:
        if rgb_image is None:
            raise ValueError("rgb_image must be provided when overlay=True")
        if rgb_image.shape[:2] != sim_color.shape[:2]:
            raise ValueError(
                "rgb_image and sim_map must have the same spatial dimensions"
            )

        # Convert both to float for blending
        sim_color = sim_color.astype(np.float32)
        rgb_image = rgb_image.astype(np.float32)
        blended = (alpha * sim_color + (1 - alpha) * rgb_image).astype(np.uint8)
        return blended

    return sim_color


def similarity_heatmap_point(
    sim_point, colormap=cv2.COLORMAP_JET, sim_scale=1.0, bgr=False
):
    """
    Generates a similarity color for a single point.

    Args:
        sim_point (np.ndarray): Similarity of point (0.0-1.0)
        colormap (int): OpenCV colormap (e.g., cv2.COLORMAP_JET)

    Returns:
        np.ndarray: RGB image (H, W, 3) visualizing similarity
    """
    sim_point = np.clip(sim_point * sim_scale, 0.0, 1.0)
    sim_point = ((1 - sim_point) * 255).astype(np.uint8)
    sim_color = cv2.applyColorMap(sim_point, colormap)
    if bgr:
        sim_color = cv2.cvtColor(sim_color, cv2.COLOR_BGR2RGB)
    return sim_color[0][0]


# create a scaled image of uint8 from a image of semantics
def labels_to_image(
    label_img,
    semantics_color_map,
    bgr=False,
    ignore_labels=[],
    overlay=False,
    alpha=0.5,
    rgb_image=None,
):
    """
    Converts a class label image to an RGB image.
    Args:
        label_img: 2D array of class labels.
        label_map: List or array of class RGB colors.
    Returns:
        rgb_output: RGB image as a NumPy array.
    """
    semantics_color_map = np.array(semantics_color_map, dtype=np.uint8)
    if bgr:
        semantics_color_map = semantics_color_map[:, ::-1]

    rgb_output = semantics_color_map[label_img]

    if len(ignore_labels) > 0 or overlay:
        if rgb_image is None:
            raise ValueError(
                "rgb_image must be provided if ignore_labels or overlay is not empty"
            )
        else:
            mask = np.isin(label_img, ignore_labels)
            rgb_output[mask] = rgb_image[mask]
            if overlay:
                # Convert to float32 for blending
                rgb_output = rgb_output.astype(np.float32)
                rgb_image = rgb_image.astype(np.float32)
                rgb_output = (alpha * rgb_output + (1 - alpha) * rgb_image).astype(
                    np.uint8
                )

    return rgb_output


def rgb_to_class(rgb_labels, label_map):
    """
    Converts an RGB label image to a class label image.
    Args:
        rgb_labels: Input RGB image as a NumPy array.
        label_map: List or array of class RGB colors.
    Returns:
        class_image: 2D array of class labels.
    """
    rgb_np = np.array(rgb_labels, dtype=np.uint8)[:, :, :3]
    label_map = np.array(label_map, dtype=np.uint8)

    reshaped = rgb_np.reshape(-1, 3)
    class_image = np.zeros(reshaped.shape[0], dtype=np.uint8)

    # Create a LUT for color matching
    for class_label, class_color in enumerate(label_map):
        matches = np.all(reshaped == class_color, axis=1)
        class_image[matches] = class_label

    return class_image.reshape(rgb_np.shape[:2])


def single_label_to_color(label, semantics_color_map, bgr=False):
    label = int(label)  # ensure label is a Python int
    color = semantics_color_map[label]
    if bgr:
        color = color[::-1]
    return color


def get_labels_color_map(semantic_dataset_type: SemanticDatasetType, **kwargs):
    if semantic_dataset_type == "voc":
        return get_voc_color_map()
    elif semantic_dataset_type == "cityscapes":
        return get_cityscapes_color_map()
    elif semantic_dataset_type == "ade20k":
        return get_ade20k_color_map()
    elif semantic_dataset_type == "nyu40":
        return get_nyu40_color_map()
    elif semantic_dataset_type == "custom_set":
        return get_coco_color_map(kwargs["num_classes"])
    elif semantic_dataset_type == "feature_similarity":
        if "num_classes" not in kwargs:
            raise ValueError(
                "num_classes must be provided if semantic_dataset_type is CUSTOM_SET"
            )
        return get_generic_color_map(kwargs["num_classes"])
    else:
        raise ValueError("Unknown dataset name: {}".format(semantic_dataset_type))


def get_labels_name(semantic_dataset_type: SemanticDatasetType):
    if semantic_dataset_type == "voc":
        return get_voc_labels()
    elif semantic_dataset_type == "cityscapes":
        return get_cityscapes_labels()
    elif semantic_dataset_type == "ade20k":
        return get_ade20k_labels()
    elif semantic_dataset_type == "nyu40":
        return get_nyu40_labels()
    elif semantic_dataset_type == "custom_set":
        raise ValueError("CUSTOM_SET does not have predefined labels")
    else:
        raise ValueError("Unknown dataset name: {}".format(semantic_dataset_type))
