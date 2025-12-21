import numpy as np

# Load the calibrated homography mapping camera pixels to pitch-map coordinates
H = np.load("project/Computer-Vision-FCH/calibration/aio_homography_cam_to_map.npy")

# Compute the inverse homography to enable pitch-to-camera back-projection
H_inv = np.linalg.inv(H)

# Persist both forward and inverse homographies for use in the analytics pipeline
np.savez(
    "project/Computer-Vision-FCH/calibration/homography.npz",
    H=H,
    H_inv=H_inv
)