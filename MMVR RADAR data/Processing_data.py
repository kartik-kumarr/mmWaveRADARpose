import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def getDir(root):
    dirNames = next(os.walk(root))[1] 
    dirNames.remove('figs')

    walking = ['d1s1', 'd1s2', 'd3s2', 'd4s1']
    jumping = [i for i in dirNames if i not in walking]

    return {'walking':walking, 'jumping': jumping}

### Function to get data from MMVR dataset
import numpy as np

def getDATA(index):
    data_files = ['radar', 'bbox', 'pose', 'mask']  # file types
    data = {}

    for file_type in data_files:
        file_path = f"{index}_{file_type}.npz"
        with np.load(file_path, mmap_mode='r') as npz:
            # Convert npz contents to dictionary
            data[file_type] = {key: npz[key] for key in npz.files}

    # Access data like this:
    hori = data['radar']['hm_hori']
    vert = data['radar']['hm_vert']
    bbox_hori = data['bbox']['bbox_hori']
    bbox_vert = data['bbox']['bbox_vert']
    bbox_i = data['bbox']['bbox_i']
    kp = data['pose']['kp']
    mask = data['mask']['mask']

    return hori, vert, bbox_hori, bbox_vert, bbox_i, kp, mask




#### Function to process each folder



def process_file(f, filePath, timeStep, folderName):
    index = os.path.join(filePath, f)
    hori, vert, bbox_hori, bbox_vert, bbox_i, kp, mask = getDATA(index)
    
    return {
        'timeStep': f"{timeStep}_{folderName}_{f}",
        'hori': np.array2string(hori, separator=',', max_line_width=np.inf, threshold=np.inf),
        'vert': np.array2string(vert, separator=',', max_line_width=np.inf, threshold=np.inf),
        'bbox_hori': np.array2string(bbox_hori, separator=',', max_line_width=np.inf, threshold=np.inf),
        'bbox_vert': np.array2string(bbox_vert, separator=',', max_line_width=np.inf, threshold=np.inf),
        'bbox_i': np.array2string(bbox_i, separator=',', max_line_width=np.inf, threshold=np.inf),
        'kp': np.array2string(kp, separator=',', max_line_width=np.inf, threshold=np.inf),
        'mask': np.array2string(mask, separator=',', max_line_width=np.inf, threshold=np.inf)
    }

def process_folder(root, folderName, outputFile):
    folder_path = os.path.join(root, folderName)
    currDirNames = next(os.walk(folder_path))[1]  
    header = False

    for timeStep in currDirNames:
        print("#################################")
        print(f"Processing timeStep: {timeStep}")
        filePath = os.path.join(folder_path, timeStep)
        allFiles = sorted(next(os.walk(filePath))[2])
        arr = np.arange(0, len(allFiles)//5, 1)
        arrSTR = np.char.zfill(arr.astype(str), 5) 
        chunkFiles = arrSTR

        ## Parallel processing
        with ThreadPoolExecutor(max_workers=12) as executor:
            data_rows = list(executor.map(lambda f: process_file(f, filePath, timeStep, folderName), chunkFiles))

        if data_rows:
            dfChunk = pd.DataFrame(data_rows)
            dfChunk.to_csv(outputFile, mode='a', index=False, header=not header)
            header = True


def createCSV(root, dirNames, CSVfileName): 
    data_frames = [] # Store DataFrames in a list 
    for folderName in dirNames: 
        process_folder(root, folderName, CSVfileName)
    return