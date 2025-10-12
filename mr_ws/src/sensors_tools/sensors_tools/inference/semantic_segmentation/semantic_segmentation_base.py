from dataclasses import dataclass
from typing import Optional
from . import semantic_types

@dataclass
class SemanticSegmentationBaseConfig:
    """
      Configuration class for Semantic Segmentation
    """

    semantic_feature_type: semantic_types.SemanticFeatureType = "probability_vector"
    """ Semantic feature type """

    device: Optional[str] = None
    """ Device to use """


class SemanticSegmentationBase:
    def __init__(self, model, transform, device, semantic_feature_type: semantic_types.SemanticFeatureType):
        self.model = model
        self.transform = transform
        self.device = device
        self.semantic_feature_type = semantic_feature_type

        self.semantics = None

    def infer(self, image):
        raise NotImplementedError
    
    # TODO(anonym): test if this works directly for transforming a single label
    def to_rgb(self, semantics, bgr=False, feature_type=None, overlay=False, rgb_image=None):
        return NotImplementedError