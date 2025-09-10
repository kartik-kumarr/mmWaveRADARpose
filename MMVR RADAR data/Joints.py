import numpy as np


connections = np.array([[13, 15], [11, 13], [14, 16], [12, 14], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]])



## COCO joints names

jointDict = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle"
}


revJointDict = {v:k for k,v in jointDict.items()}


## making a vertex mid and corresponding vertex for making angle calculation easier

######################## Shoulder ####################################
shoulderRight = ["Right Elbow","Right Shoulder","Right Hip"]
shoulderLeft = ["Left Elbow","Left Shoulder","Left Hip"]


####################### Elbow ##################################################
elbowRight = ["Right Wrist", "Right Elbow", "Right Shoulder"]
elbowLeft = ["Left Wrist", "Left Elbow", "Left Shoulder"]


####################### Hip ###################################################]
hipRight = ["Right Shoulder", "Right hip", "Right knee"]
hipLeft = ["Left Shoulder", "Left hip", "Left knee"]


############################## knee ###########################################
kneeRight = ["Right Ankle", "Right knee", "Right hip"]
kneeLeft = ["Left Ankle", "Left knee", "Left hip"]


angleTriplets = {
    "Right Shoulder": ["Right Elbow", "Right Shoulder", "Right Hip"],
    "Left Shoulder": ["Left Elbow", "Left Shoulder", "Left Hip"],
    "Right Elbow": ["Right Wrist", "Right Elbow", "Right Shoulder"],
    "Left Elbow": ["Left Wrist", "Left Elbow", "Left Shoulder"],
    "Right Hip": ["Right Shoulder", "Right Hip", "Right Knee"],
    "Left Hip": ["Left Shoulder", "Left Hip", "Left Knee"],
    "Right Knee": ["Right Ankle", "Right Knee", "Right Hip"],
    "Left Knee": ["Left Ankle", "Left Knee", "Left Hip"]
}


######################################### Angle Calulation ################################

def calculateAngle(a, b, c):
    """
    Calculate the angle formed by three points (a, b, c)
    where b is the mid connecting vertex (e.g., the knee).
    """

    # Extract coordinates from the Normalized Landmark objects
    a_coords = np.array(a)
    b_coords = np.array(b)
    c_coords = np.array(c)

    # Create vectors AB and BC
    vector_ab = a_coords - b_coords
    vector_bc = c_coords - b_coords

    # Compute the dot product
    dot = np.dot(vector_ab, vector_bc)

    # Compute the magnitudes of the vectors
    norm_ab = np.linalg.norm(vector_ab)
    norm_bc = np.linalg.norm(vector_bc)

    if norm_ab == 0 or norm_bc == 0:
        return np.nan

    # Calculate the angle in radians and then convert to degrees
    cos_angle = np.clip(dot / (norm_ab * norm_bc), -1.0, 1.0)
    theta = np.arccos(cos_angle)
    angle = np.degrees(theta)

    return angle