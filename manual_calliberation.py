import numpy as np

fx, fy = 1000.0, 1000.0  
cx, cy = 960.0, 540.0    

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

np.savez("MyCalibration.npz", Camera_matrix=camera_matrix, distCoeff=np.zeros(5))        