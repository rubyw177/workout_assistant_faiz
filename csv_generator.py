import os, csv, cv2, random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

# # --- Dictionary that maps from joint names to keypoint indices.
# KEYPOINT_DICT = {
#     'nose': 0,
#     'left_eye': 1,
#     'right_eye': 2,
#     'left_ear': 3,
#     'right_ear': 4,
#     'left_shoulder': 5,
#     'right_shoulder': 6,
#     'left_elbow': 7,
#     'right_elbow': 8,
#     'left_wrist': 9,
#     'right_wrist': 10,
#     'left_hip': 11,
#     'right_hip': 12,
#     'left_knee': 13,
#     'right_knee': 14,
#     'left_ankle': 15,
#     'right_ankle': 16
# }

# --- List of keypoint edges
KEYPOINT_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)
]

# Add header to csv file
header = [
    "noseX",
    "noseY",
    "left_eyeX",
    "left_eyeY",
    "right_eyeX",
    "right_eyeY",
    "left_earX",
    "left_earY",
    "right_earX",
    "right_earY",
    "left_shoulderX",
    "left_shoulderY",
    "right_shoulderX",
    "right_shoulderY",
    "left_elbowX",
    "left_elbowY",
    "right_elbowX",
    "right_elbowY",
    "left_wristX",
    "left_wristY",
    "right_wristX",
    "right_wristY",
    "left_hipX",
    "left_hipY",
    "right_hipX",
    "right_hipY",
    "left_kneeX",
    "left_kneeY",
    "right_kneeX",
    "right_kneeY",
    "left_ankleX",
    "left_ankleY",
    "right_ankleX",
    "right_ankleY",
    "exercise"
]

# ---Empty list for keypoints data and labels
data = [header]
labels_dict = {"push up": 0,
               "squat": 1}

# --- Define a function to predict keypoints from image using MoveNet TFHub
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

def predict_keypoints(model, image_path):
    # Read image files and rersize it to 192x192
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

    # Predict using movenet
    movenet = model.signatures["serving_default"]
    outputs = movenet(image)

    # Return keypoints in (1, 1, 17, 3)
    keypoints = outputs["output_0"]

    return image, keypoints

# Display the keypoints with the image
def display_keypoints(input_image, keypoints, width=512, height=512):
    input_image = tf.image.resize_with_pad(input_image, width, height)
    input_image = tf.cast(input_image, dtype=tf.uint8)

    npy_image = np.squeeze(input_image.numpy(), axis=0)
    npy_image = cv2.resize(npy_image, (width, height))
    npy_image = cv2.cvtColor(npy_image, cv2.COLOR_RGB2BGR)

    for keypoint in keypoints[0][0]:
        x = int(keypoint[1] * width)
        y = int(keypoint[0] * height)
        cv2.circle(npy_image, (x, y), 4, (0, 255, 255), -1)

    for edge in KEYPOINT_EDGES:
        x1 = int(keypoints[0][0][edge[0]][1] * width)
        y1 = int(keypoints[0][0][edge[0]][0] * height)
        x2 = int(keypoints[0][0][edge[1]][1] * width)
        y2 = int(keypoints[0][0][edge[1]][0] * height)
        cv2.line(npy_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Pose Estimation", npy_image)
    cv2.waitKey()

# --- Iterate through dataset image files and save prediction in the empty lists
root_dir = "C:\\Users\\willi\\Desktop\\workout assistant\\dataset"

# random_list = [100, 200, 50, 400, 300, 150]

for folder in os.listdir(root_dir):

    # Assign label according to current folder and join the file path
    label = labels_dict[str(folder)]
    current_folder = os.path.join(root_dir, folder)

    counter = 0
    for file in os.listdir(current_folder):
        # Reset list every file iteration
        data_temp = []

        # Predict current image with movenet
        current_image = os.path.join(root_dir, folder, file)
        input_image, keypoints = predict_keypoints(model, current_image)

        # Add every part of body keypoint coordinates and the label to a list
        for keypoint in keypoints[0][0]:
            x = float(keypoint[1].numpy())
            y = float(keypoint[0].numpy())
            data_temp.append(x)
            data_temp.append(y)
            print("\n[!] Keypoints output:")
            print(keypoint)
            print(x, y)
            print(len(data_temp))
        data_temp.append(label)

        # Append the features and labels row to the list
        data.append(data_temp)

        # Visualize image and overlay keypoints plot
        # if counter in random_list:
        #     display_keypoints(input_image, keypoints, 400, 400)

        counter += 1
    
    print("[!] Data in {} folder have been added to the list!".format(folder))

# print("\n")
# print(data)
# print("\n")

# --- Add data to a csv file
# Specify csv file path
csv_file_path = os.path.join(root_dir, "keypoints_data.csv")

# Open in write mode
with open(csv_file_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write each line in data to csv file
    for row in data:
        writer.writerow(row)
    
    print("[!] All rows have been added!")





        







