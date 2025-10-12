import numpy as np

def pcd_from_rgb_depth(rgb: np.ndarray, depth: np.ndarray, cx: int, cy: int, fx: float, fy: float, stride: int =1):
    """A function that converts an rgb image and a depth image to a point cloud
    Depth image is measured with respect to image plane. If measured from camera center uncomment lines
    Depth is along X axis
    Args:
        rgb: a numpy array with shape (H, W, 3)
        depth: a numpy array with shape (H, W)
        cx: x coordinate of the principal point
        cy: y coordinate of the principal point
        fx: focal length in x direction
        fy: focal length in y direction
        stride: stride to downsample the point cloud
    Returns:
        pcd: a point cloud as a numpy array with shape (H*W, 4)
    """
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255
    H = depth.shape[0]
    W = depth.shape[1]
    columns, rows = np.meshgrid(np.linspace(0, W-1, num=int(W/stride)), np.linspace(0, H-1, num=int(H/stride)))
    point_depth = depth[::stride, ::stride]
    y = -(columns - cx) * point_depth / fx # Originally x : Now -y
    z = -(rows - cy) * point_depth / fy # Originally y : Now -z
    x = point_depth # Originally z : Now x
    pcd = np.dstack((x, y, z)).astype(np.float32)
    colors = rgb[::stride, ::stride, :] * 255
    colors = colors.astype(np.uint8)
    # Add alpha channel
    colors = np.dstack((colors, np.ones((colors.shape[0], colors.shape[1], 1), dtype=np.uint8) * 255))
    return pcd, colors