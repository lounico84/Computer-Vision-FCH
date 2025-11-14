import numpy as np

H = np.load("project/Computer-Vision-FCH/calibration/aio_homography_cam_to_map.npy")
H_inv = np.linalg.inv(H)

np.savez("project/Computer-Vision-FCH/calibration/homography.npz", H=H, H_inv=H_inv)