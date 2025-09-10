import os
import numpy as np
import pandas as pd

def getDir(root):
    dirNames = next(os.walk(root))[1] 
    dirNames.remove('figs')

    walking = ['d1s1', 'd1s2', 'd3s2', 'd4s1']
    jumping = [i for i in dirNames if i not in walking]

    return {'walking':walking, 'jumping': jumping}

### Function to get data from MMVR dataset

def getDATA(index):
    ## Loading meta data
    with np.load(index + '_meta.npz', mmap_mode='r') as meta_data:
        pass

    ## Loading radar data from npz files
    with np.load(index + '_radar.npz', mmap_mode='r') as radar_data:
        hori = radar_data['hm_hori']  # (256, 128)
        vert = radar_data['hm_vert']  # (256, 128)

    ## Loading bounding box data from npz files
    with np.load(index + '_bbox.npz', mmap_mode='r') as bbox_data:
        bbox_i = bbox_data['bbox_i']
        bbox_hori = bbox_data['bbox_hori']
        bbox_vert = bbox_data['bbox_vert']

    ## Loading pose data from npz files  from npz files
    with np.load(index + '_pose.npz', mmap_mode='r') as pose_data:
        kp = pose_data['kp']  # (n, 17, 3)

    ## Loading segmentation mask data from npz files
    with np.load(index + '_mask.npz', mmap_mode='r') as mask_data:
        mask = mask_data['mask']

    return hori, vert, bbox_hori, bbox_vert, bbox_i, kp, mask




#### Function to process each folder


def process_folder(root, folderName, outputFile):
    folder_path = os.path.join(root, folderName)
    print(folder_path)

    # Get the immediate subdirectories (timeSteps)
    currDirNames = next(os.walk(folder_path))[1]  


    header = False

    # Process each timeStep
    for timeStep in currDirNames:
        filePath = os.path.join(folder_path, timeStep)
        print("#################################")
        print(f"Processing timeStep: {timeStep}")

        allFiles = sorted(next(os.walk(filePath))[2])

        # print(next(os.walk(filePath))[2])

        # print(len(allFiles)/5)
        # break

        

        arr = np.arange(0, len(allFiles)//5, 1)
        arrSTR = np.char.zfill(arr.astype(str), 5) 
        chunkFiles = arrSTR
        data_rows = []
        for f in chunkFiles:
            index = os.path.join(filePath, f)
            # print(index)
            hori, vert, bbox_hori, bbox_vert, bbox_i, kp, mask = getDATA(index)

            # Append the extracted data to the list
            data_rows.append({
                'timeStep': f"{timeStep}_{folderName}_{f}",
                'hori': hori,
                'vert': vert,
                'bbox_hori': bbox_hori,
                'bbox_vert': bbox_vert,
                'bbox_i': bbox_i,
                'kp': kp,
                'mask': mask
                    })

            # ## Incrementing the timeStepDir name
            # startDir = f"{int(startDir) + 1:05d}"

        ## Yield a DataFrame for each batch
        if data_rows:
            dfChunk = pd.DataFrame(data_rows)
            dfChunk.to_csv(outputFile, mode = 'a', index = False, header=not header)
            header = True
    return  