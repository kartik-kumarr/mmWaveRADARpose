import os
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 

def getDir(root):
    dirNames = next(os.walk(root))[1] 
    if 'figs' in dirNames:
        dirNames.remove('figs')

    walking = ['d1s1', 'd1s2', 'd3s2', 'd4s1']
    jumping = [i for i in dirNames if i not in walking]

    return {'walking': walking, 'jumping': jumping}


def getDATA(index):
    data_files = ['meta', 'radar', 'bbox', 'pose', 'mask']
    data = {}
    for file_type in data_files:
        file_path = f"{index}_{file_type}.npz"
        with np.load(file_path, mmap_mode='r') as npz:
            data[file_type] = {key: npz[key] for key in npz.files}

    ID = data['meta']['global_frame_id']
    hori = data['radar']['hm_hori'].astype('float32')
    vert = data['radar']['hm_vert'].astype('float32')
    kp = data['pose']['kp'].astype('float32')
    bbox_hori = data['bbox']['bbox_hori'].astype('float32')
    bbox_vert = data['bbox']['bbox_vert'].astype('float32')
    bbox_i = data['bbox']['bbox_i'].astype('float32')
    mask = data['mask']['mask'].astype('float32')

    return ID, hori, vert, bbox_hori, bbox_vert, bbox_i, kp, mask


def process_file(f, filePath):
    index = os.path.join(filePath, f)
    return getDATA(index)


def process_folder_h5(root, folderName, h5f, max_workers=12):
    folder_path = os.path.join(root, folderName)
    # print(f"Processing folder: {folderName}")

    # List timesteps
    timesteps = sorted(next(os.walk(folder_path))[1])

    for timestep in tqdm(timesteps, desc=f"Processing {folderName} timesteps"):
        # print(f"Processing timestep: {timestep}")
        timestep_path = os.path.join(folder_path, timestep)
        all_files = sorted(next(os.walk(timestep_path))[2])

        # Only take unique indices (assuming 5 files per index)
        num_frames = len(all_files) // 5
        indices = np.char.zfill(np.arange(num_frames).astype(str), 5)

        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            data_list = list(executor.map(lambda f: process_file(f, timestep_path), indices))

        IDs, horis, verts, bbox_horis, bbox_verts, bbox_is, kps, masks = zip(*data_list)

        ## Converting it to numpy arrays
        IDs = np.array(IDs)
        horis = np.stack(horis)
        verts = np.stack(verts)
        bbox_horis = np.stack(bbox_horis)
        bbox_verts = np.stack(bbox_verts)
        bbox_is = np.stack(bbox_is)
        kps = np.stack(kps)
        masks = np.stack(masks)

        n_new = len(IDs)
        for dset, arr in zip(
            [h5f['ID'], h5f['hori'], h5f['vert'], h5f['bbox_hori'], h5f['bbox_vert'], h5f['bbox_i'], h5f['kp'], h5f['mask']],
            [IDs, horis, verts, bbox_horis, bbox_verts, bbox_is, kps, masks]
        ):
            dset.resize(dset.shape[0] + n_new, axis=0)
            dset[-n_new:] = arr


def createH5(root, dirNames, h5_file):
    H, W = 256, 128
    num_joints = 17

    with h5py.File(h5_file, 'w') as h5f:
        ## Initialize dataset 
        h5f.create_dataset('ID', shape=(0,), maxshape=(None,), chunks=True, compression='gzip', dtype = h5py.string_dtype(encoding='utf-8'))
        h5f.create_dataset('hori', shape=(0, H, W), maxshape=(None, H, W), chunks=True, compression='gzip', dtype='float32')
        h5f.create_dataset('vert', shape=(0, H, W), maxshape=(None, H, W), chunks=True, compression='gzip', dtype='float32')
        h5f.create_dataset('bbox_hori', shape=(0, 1, 4), maxshape=(None, 1, 4), chunks=True, compression='gzip', dtype='float32')
        h5f.create_dataset('bbox_vert', shape=(0, 1, 4), maxshape=(None, 1, 4), chunks=True, compression='gzip', dtype='float32')
        h5f.create_dataset('bbox_i', shape=(0, 1, 5), maxshape=(None, 1, 5), chunks=True, compression='gzip', dtype='float32')
        h5f.create_dataset('kp', shape=(0, 1, num_joints, 3), maxshape=(None, 1, num_joints, 3), chunks=True, compression='gzip', dtype='float32')
        h5f.create_dataset('mask', shape=(0, 1, 480, 640), maxshape=(None, 1, 480, 640), chunks=True, compression='gzip', dtype='float32')

        # Process all folders
        for folderName in tqdm(dirNames, desc=f"Processing {dirNames} folder"):
            process_folder_h5(root, folderName, h5f)
