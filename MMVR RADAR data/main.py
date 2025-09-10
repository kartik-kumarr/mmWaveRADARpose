import numpy as np
import os
import pandas as pd
import cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Processing_data import getDir, process_folder



## Loading the dataset folder
root = 'c:/Users/karti/Downloads/Datasets'
dirNames = getDir(root)

print(dirNames['walking'])






## function to process the walking posture data store data in walking.csv
def main(root, dirNames):
    data_frames = []  # Store DataFrames in a list

    for folderName in dirNames:
        process_folder(root, folderName,  "Walking.csv")
    return

ProcessWalking = main(root, dirNames['walking'])