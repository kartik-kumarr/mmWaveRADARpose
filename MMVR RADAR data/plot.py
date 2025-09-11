import numpy as np
import cv2

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from Joints import connections, jointDict

def plotHeatmapsWithMask(hori, vert, mask, bbox_i, kp, connections):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    scale_x = 640 / 128  # Width scale
    scale_y = 480 / 256  # Height scale
    hori_resized = cv2.resize(hori, (640, 480), interpolation=cv2.INTER_LINEAR)
    vert_resized = cv2.resize(vert, (640, 480), interpolation=cv2.INTER_LINEAR)

    # # Normalize heatmaps for better visualization
    # hori_resized = (hori_resized - np.min(hori_resized)) / (np.max(hori_resized) - np.min(hori_resized))
    # vert_resized = (vert_resized - np.min(vert_resized)) / (np.max(vert_resized) - np.min(vert_resized))

    ## Create an RGB mask visualization
    mask_vis = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    mask_vis[mask[0] == 1] = (255, 0, 0)  # Apply red for segmentation mask

    ## Overlay horizontal heatmap
    axes[0].imshow(mask_vis)  # Show segmentation mask
    axes[0].imshow(hori_resized, cmap='jet', alpha=0.5)  # Overlay heatmap with transparency
    axes[0].set_title("Horizontal Heatmap over Segmentation")

    ## Overlay vertical heatmap
    axes[1].imshow(mask_vis)
    axes[1].imshow(vert_resized, cmap='jet', alpha=0.5)
    axes[1].set_title("Vertical Heatmap over Segmentation")

    plt.show()
    return 



def plot_heatmaps_and_annotations(hori, vert, mask, bbox_i, kp, connections):
    # Plot horizontal and vertical heatmaps side by side
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(hori)
    ax.set_title('Horizontal Heatmap')

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(vert)
    ax.set_title('Vertical Heatmap')

    plt.show()

    # Plot segmentation, bounding boxes, and keypoints
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Create an empty image to overlay the masks
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # print(mask.shape)


    for j in range(mask.shape[0]):
        ## converting the mask data from boolean to int
        # tempMask = mask[j, :, :].astype(np.uint8)
        # ## resizing the mask to 240*320 dimensions
        # tempMask = cv2.resize(tempMask, (128, 256), interpolation = cv2.INTER_NEAREST)

        # tempMask = np.expand_dims(tempMask, axis=0)

        # Apply segmentation mask (green)
        img[mask[j, :, :]==1] = (0, 255, 0)

        # # Scaling factors
        # scaleX = 128 / mask.shape[2]
        # scaleY = 256 / mask.shape[1]


        # Draw bounding boxes (red)
        x1, y1, x2, y2 = bbox_i[j, :4]
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((int(x1), int(y1)), width, height, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # Plot keypoints (red Gaussian circles)
        for connection in connections:
          for idx in connection:
            x = float(kp[j, idx, 0])
            y = float(kp[j, idx, 1])
            z = float(kp[j, idx, 2])
            # ellipse = Ellipse((x, y), width=20, height=20, edgecolor='blue', facecolor='red', alpha=0.5)
            # ax.add_patch(ellipse)
          x = kp[j, connection, 0]
          y = kp[j, connection, 1]
        #     z = kp[j, connection, 2]
          ax.plot(x, y, color='b', marker='.')




        # # # # Plot keypoints (blue)
        # for connection in connections:
        #     x = kp[j, connection, 0]
        #     y = kp[j, connection, 1]
        #     z = kp[j, connection, 2]

        #     # if jointDict[connection[0]] in angleTriplets:
        #     #   print(computeAngle(connection[0], j, kp))
        #     ellipse = Ellipse((x, y), width=0.1, height=0.1, edgecolor='red', facecolor='red', alpha=0.3)
        #     ax.add_patch(ellipse)
        #     # ax.plot(x, y, z, color='b', marker='.',  linestyle='None')

    ax.imshow(img)
    ax.set_title('Segmentation, Bounding Boxes, and Keypoints (2D)')
    plt.show()


