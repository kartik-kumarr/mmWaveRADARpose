import os
import numpy as np
import h5py
from tqdm import tqdm
from skimage.transform import resize
import cv2
from concurrent.futures import ThreadPoolExecutor

def checKNans(heatmap):
    y, x = np.where(np.isnan(heatmap))
    return list(zip(x, y))

def flipHeatmaps(orig, toBeFilled, Nanidx):
    toBeFilled_filled = toBeFilled.copy()
    for x, y in Nanidx:
        toBeFilled_filled[y, x] = orig[y, x]
    return toBeFilled_filled

def fillWithNeigh(heatmap):
    mask = np.isnan(heatmap).astype(np.uint8) * 255
    heatmap32 = np.nan_to_num(heatmap, nan=0).astype(np.float32)  # Use float32 to save memory
    heatmapFilled = cv2.inpaint(heatmap32, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return heatmapFilled

def flipAndFill(horizontal, vertical):
    NanMaskHori = checKNans(horizontal)
    NanMaskVert = checKNans(vertical)
    if len(NanMaskVert) > 0 and len(NanMaskHori) == 0:
        vertical = flipHeatmaps(horizontal, vertical, NanMaskVert)
        return horizontal, vertical
    elif len(NanMaskVert) == 0 and len(NanMaskHori) > 0:
        horizontal = flipHeatmaps(vertical, horizontal, NanMaskHori)
        return horizontal, vertical
    elif len(NanMaskVert) > 0 and len(NanMaskHori) > 0:
        filledHori = fillWithNeigh(horizontal)
        filledVert = fillWithNeigh(vertical)
        return filledHori, filledVert
    else:
        return horizontal, vertical

## normalizes the heatmaps
def normalizeHeatmaps(heatmap):
    return cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


## Function to process the heatmaps

def preprocessHeatmaps(filepath, newShape, chunkSize, n_threads):
    with h5py.File(filepath, 'r+') as data:
        n_Frames = data['hori'].shape[0]

        # Create datasets for processed heatmaps if they don't exist
        if 'filled_hori' not in data:
            data.create_dataset('filled_hori', shape=(n_Frames, *newShape),
                                dtype='float32', chunks=True, compression='gzip')
            data.create_dataset('filled_vert', shape=(n_Frames, *newShape),
                                dtype='float32', chunks=True, compression='gzip')

        # Process in chunks
        for i in tqdm(range(0, n_Frames, chunkSize), desc="Preprocessing the Heatmaps"):
            lastFrame = min(i + chunkSize, n_Frames)
            hori_chunk = data['hori'][i:lastFrame]
            vert_chunk = data['vert'][i:lastFrame]

            # Define a function to process a single frame
            def process_frame(frame_pair):
                hori, vert = frame_pair
                # Fill NaNs
                hori, vert = flipAndFill(hori, vert)
                # Normalize
                hori = normalizeHeatmaps(hori)
                vert = normalizeHeatmaps(vert)
                # Resize
                hori_resized = resize(hori, newShape, order=3, preserve_range=True, anti_aliasing=True)
                vert_resized = resize(vert, newShape, order=3, preserve_range=True, anti_aliasing=False)
                return hori_resized.astype('float32'), vert_resized.astype('float32')

            # Process the chunk in parallel
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                results = list(executor.map(process_frame, zip(hori_chunk, vert_chunk)))

            # Split results and write back to HDF5
            hori_resized_chunk, vert_resized_chunk = zip(*results)
            data['filled_hori'][i:lastFrame] = np.stack(hori_resized_chunk)
            data['filled_vert'][i:lastFrame] = np.stack(vert_resized_chunk)

            # Deleting the garbage
            del hori_chunk, vert_chunk, results, hori_resized_chunk, vert_resized_chunk



############################### Function to preprocess keypoints #################################################################

def preProcessY_chunk(kp_chunk, bbox_chunk, mask_chunk, new_height, new_width):
    """
    Processes a chunk of keypoints, bounding boxes, and masks.
    kp_chunk: (chunk_size, num_joints, 3)
    bbox_chunk: (chunk_size, 5)
    mask_chunk: (chunk_size, H, W)
    Returns: processed kp, bbox, mask
    """
    n_frames = kp_chunk.shape[0]
    num_joints = kp_chunk.shape[1]

    processed_kp = np.zeros_like(kp_chunk, dtype=np.float32)
    processed_bbox = np.zeros_like(bbox_chunk, dtype=np.int32)
    processed_mask = np.zeros((n_frames, new_height, new_width), dtype=np.uint8)
   

    ##Scaling factors
    scaleX = new_width / mask_chunk.shape[2]
    scaleY = new_height / mask_chunk.shape[1]

    for j in range(n_frames):
        ## Resizing the mask
        processed_mask[j] = cv2.resize(mask_chunk[j].astype(np.uint8), 
                                       (new_width, new_height), 
                                       interpolation=cv2.INTER_NEAREST)

        ## Scaling bbox
        processed_bbox[j, :4] = (bbox_chunk[j, :4] * np.array([scaleX, scaleY, scaleX, scaleY])).astype(int)
        processed_bbox[j, 4] = bbox_chunk[j, 4]  # copy remaining value if any

        ## Scaling keypoints
        processed_kp[j] = kp_chunk[j]
        processed_kp[j, :, 0] *= scaleX
        processed_kp[j, :, 1] *= scaleY


    return processed_kp, processed_bbox, processed_mask


def preprocessH5_Ytrain(h5_file_path, new_mask_shape=(65, 65), chunk_size=1000):
    """
    Processes kp, bbox_i, mask datasets in HDF5 file in chunks.
    Returns processed datasets (as HDF5 or in-memory arrays).
    """
    with h5py.File(h5_file_path, 'r+') as h5f:
        n_frames = h5f['kp'].shape[0]
        num_joints = h5f['kp'].shape[2]

        ## Create new datasets for processed data 
        if 'kp_scaled' not in h5f:
            h5f.create_dataset('kp_scaled', shape=(n_frames, num_joints, 3),
                               dtype='float32', chunks=True, compression='gzip')
        if 'bbox_scaled' not in h5f:
            h5f.create_dataset('bbox_scaled', shape=(n_frames, 5),
                               dtype='int32', chunks=True, compression='gzip')
        if 'mask_scaled' not in h5f:
            h5f.create_dataset('mask_scaled', shape=(n_frames, *new_mask_shape),
                               dtype='uint8', chunks=True, compression='gzip')

        ## Processing in chunks
        for i in range(0, n_frames, chunk_size):
            last = min(i + chunk_size, n_frames)

            kp_chunk = h5f['kp'][i:last, 0, :, :]      
            bbox_chunk = h5f['bbox_i'][i:last, 0, :]    
            mask_chunk = h5f['mask'][i:last, 0, :, :]   

            kp_scaled, bbox_scaled, mask_scaled = preProcessY_chunk(
                kp_chunk, bbox_chunk, mask_chunk, new_mask_shape[0], new_mask_shape[1]
            )

            ## Saving the data
            h5f['kp_scaled'][i:last] = kp_scaled
            h5f['bbox_scaled'][i:last] = bbox_scaled
            h5f['mask_scaled'][i:last] = mask_scaled
           
    return 




##### Funtion to combine the temporal data        ####

