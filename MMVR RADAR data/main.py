import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast
## imporing custom functions
from Processing_data import getDir, createCSV
from Joints import connections
from plot import plot_heatmaps_and_annotations




## Loading the dataset folder
root = 'c:/Users/karti/Downloads/Datasets'
dirNames = getDir(root)

# print(dirNames['walking'])

## Calling function to create two CSV for postures a) Walking, b) Jumping

ProcessWalking = createCSV(root, dirNames['walking'], 'walking.csv')
ProcessJumping = createCSV(root, dirNames['jumping'], 'jumping.csv')


# fig, ax = plt.subplots()   #

# # ## Ploting to check if walking posture is loaded correctly
# df = pd.read_csv('walking.csv')
# print(df.dtypes)

# df =  pd.to_numeric(df.iloc[:, 1:])
# print(df.dtypes)

# # List of heatmap/array columns
# array_cols = ['hori', 'vert', 'mask', 'kp', 'bbox_hori', 'bbox_vert', 'bbox_i']

# # Parse each array column from string to NumPy array
# for col in array_cols:
#     df[col] = df[col].to_numpy(dtype=np.float16)

# # If you want, convert numeric columns to float
# numeric_cols = [c for c in df.columns if c not in array_cols + ['timeStep']]
# df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# # Now you can access NumPy arrays for heatmaps
# hori_arrays = np.array(df['hori'].to_list(), dtype=object)  

# # timeStep, hori, vert, bbox_hori, bbox_vert, bbox_i, kp, mask = dfWalking.iloc[3, :]
# # print(hori.shape)

# # dfWalking.iloc[:, 1:] = dfWalking.iloc[:, 1:].applymap(
# #     lambda x: np.array(ast.literal_eval(x), dtype=np.float16)
# # )


# print(type(dfWalking.iloc[0,1]))  # <class 'numpy.ndarray'>
# print(dfWalking.iloc[0,1].shape)  # shape of the heatmap

# # timeStep, hori, vert, bbox_hori, bbox_vert, bbox_i, kp, mask = df[3, :]

# print(hori.shape)


# # plot_heatmaps_and_annotations(hori, vert, mask, bbox_i, kp, connections)