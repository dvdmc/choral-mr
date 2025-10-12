from dataclasses import dataclass
from typing import List, Literal
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation

@dataclass
class BaseBridgeConfig:
    """
        Base class for bridge configuration
    """

    data_types: List[Literal["rgb", "depth", "semantic", "pose"]]
    """
        List of data types that the bridge will provide
        Each bridge can specialize on this
    """

class BaseBridge(ABC):
    """
        Any bridge class should implement this interface
    """
    def __init__(self, cfg):
        """

        """
        self.cfg = cfg

    @abstractmethod
    def setup(self):
        """
            Setup the bridge
        """
        pass

    @abstractmethod
    def get_data(self) -> dict:
        """
            Get data from the bridge
        """
        pass

    def move(self) -> bool:
        """
            Apply movement to the bridge asuming 
            it will control the movement (dataset case)
        """
        raise NotImplementedError("Move not implemented for this bridge")
    
    def move_to_pose(self, traslation: np.ndarray, rotation: Rotation) -> bool:
        """
            Apply a movement to the bridge
            This function should be used when the bridge
            controls the movement of the sensor as in
            sequence datasets
        """
        raise NotImplementedError("Move to pose not implemented for this bridge")