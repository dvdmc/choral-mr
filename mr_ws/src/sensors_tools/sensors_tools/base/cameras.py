import numpy as np

from dataclasses import dataclass

@dataclass
class CameraData:
    """
        Stores camera parameters in a unified way
    """

    width: int
    """ Image width """

    height: int
    """ Image height """

    fx: float
    """ Focal length in x """

    fy: float
    """ Focal length in y """

    cx: float
    """ Principal point in x """

    cy: float
    """ Principal point in y """

    k1: float = 0.0
    """ Radial distortion coefficient k1 """

    k2: float = 0.0
    """ Radial distortion coefficient k2 """

    k3: float = 0.0
    """ Radial distortion coefficient k3 """

    p1: float = 0.0
    """ Tangential distortion coefficient p1 """

    p2: float = 0.0
    """ Tangential distortion coefficient p2 """
    
    @classmethod
    def from_fov_h(cls, width: int, height: int, fov_h: float) -> "CameraData":
        """
            Create camera info from fov_h

            Args:
                width: Image width
                height: Image height
                fov_h: Horizontal field of view

            Returns:
                CameraData: Camera info
        """
        cx = float(width) / 2
        cy = float(height) / 2
        fov_h_rad = fov_h * np.pi / 180.0
        fx = cx / (np.tan(fov_h_rad / 2))
        fy = fx * height / width
        return cls(width, height, fov_h, fx, fy, cx, cy)
    
    def __str__(self):
        return f"width: {self.width}, height: {self.height}, fx: {self.fx}, fy: {self.fy}, cx: {self.cx}, cy: {self.cy}, k1: {self.k1}, k2: {self.k2}, k3: {self.k3}, p1: {self.p1}, p2: {self.p2}"