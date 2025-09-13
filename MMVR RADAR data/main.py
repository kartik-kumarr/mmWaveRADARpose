import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast
import h5py
## imporing custom functions
from ConvertH5 import getDir,  createH5
from Processing_data import preprocessHeatmaps, preprocessH5_Ytrain
from Joints import connections
from plot import plot_heatmaps_and_annotations



# ## Loading the dataset folder
# root = 'c:/Users/karti/Downloads/Datasets'
# dirNames = getDir(root)

# print(f"Walking posture data folder names: {dirNames['walking']}")

# print(f"Jumping posture data folder names: {dirNames['jumping']}")

## Calling function to create two CSV for postures a) Walking, b) Jumping

# ProcessWalking = createH5(root, dirNames['walking'], 'walking.h5')
# ProcessJumping = createH5(root, dirNames['jumping'], 'jumping.h5')

WalkingPose = "c:/Users/karti/Downloads/Datasets/walking.h5"
JumpingPose = "c:/Users/karti/Downloads/Datasets/jumping.h5"

## Plotting example pose for walking data

# with h5py.File(WalkingPose, 'r') as data:
#     i = 100
#     print("Keys in the root directory:", list(data.keys()))
#     hori = data['hori'][i]
#     vert = data['vert'][i]
#     mask = data['mask'][i]
#     kp = data['kp'][i]
#     bbox_i = data['bbox_i'][i]
#     plot_heatmaps_and_annotations(hori, vert, mask, bbox_i, kp, connections)


## Plotting example pose for jumping data

# with h5py.File(JumpingPose, 'r') as data:
#     i = 100
#     print("Keys in the root directory:", list(data.keys()))
#     hori = data['hori'][i]
#     vert = data['vert'][i]
#     mask = data['mask'][i]
#     kp = data['kp'][i]
#     bbox_i = data['bbox_i'][i]
#     plot_heatmaps_and_annotations(hori, vert, mask, bbox_i, kp, connections)


# preprocessHeatmaps(WalkingPose, newShape=(65, 65), chunkSize=500, n_threads=12)

# preprocessHeatmaps(WalkingPose, newShape=(65, 65), chunkSize=500, n_threads=12)


# preprocessH5_Ytrain(WalkingPose, (65, 65))


with h5py.File(WalkingPose, 'r') as data:
    print("Keys in the root directory:", list(data.keys()))