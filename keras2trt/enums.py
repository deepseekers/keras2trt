from enum import Enum


class ModelObjective(Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


class OnnxOpset(Enum):
    CLASSIFICATION = 13
    DETECTION = 13
    SEGMENTATION = 15
