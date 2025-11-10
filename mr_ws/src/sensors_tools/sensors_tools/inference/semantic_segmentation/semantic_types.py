from typing import Literal

SemanticFeatureType = Literal["label", "probability_vector", "feature_vector"]
SemanticEntityType = Literal["point, object"]
SemanticDatasetType = Literal[
    "cityscapes", "ade20k", "voc", "nyu40", "feature_similarity", "custom_set"
]
SemanticSegmentationMethods = Literal["deep_lab_v3", "segformer", "clip", "trident"]
