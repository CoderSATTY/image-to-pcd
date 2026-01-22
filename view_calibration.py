import numpy as np

npz_file = 'CalibrationMatrix_college_cpt.npz' 

with np.load(npz_file) as data:
    for item in data.files:
        print(f"\n--- Array Name: '{item}' ---")
        print(f"Shape: {data[item].shape}")
        print(f"Data Type: {data[item].dtype}")
        print(f"Contents:\n{data[item]}")
