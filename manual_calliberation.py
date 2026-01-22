import numpy as np

"""
Logitech C270 Intrinsic Parameters
Resolution: 640 x 480
Diagonal FoV: 55°

Computed using:
  D = sqrt(640² + 480²) = 800 px
  f = (D/2) / tan(dFoV/2) = 400 / tan(27.5°) ≈ 768 px
"""

# Computed from 55° diagonal FoV
fx = 768.0
fy = 768.0

# Principal point at image center
cx = 320.0
cy = 240.0

print(f"Logitech C270 @ 640x480:")
print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

np.savez("MyCalibration.npz", Camera_matrix=camera_matrix, distCoeff=np.zeros(5))
print("Saved: MyCalibration.npz")