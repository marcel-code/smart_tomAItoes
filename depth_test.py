import json
import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib.colors import rgb_to_hsv

from src.settings import DATA_PATH
from src.utils.depth import PointCloudCreator

# point to the image files
image = "D_b14ffae7"  # B_93dd57ad B_0328ab97 A_1a111b40 D_b14ffae7

substrate_level = {
    "A": 10,
    "B": 7,
    "C": 10,
    "D": 10,
}

color_dict_HSV = {
    "black": [[180, 255, 30], [0, 0, 0]],
    "white": [[180, 18, 255], [0, 0, 231]],
    "red1": [[180, 255, 255], [159, 50, 70]],
    "red2": [[9, 255, 255], [0, 50, 70]],
    "green": [[89, 255, 255], [36, 50, 70]],
    "blue": [[128, 255, 255], [90, 50, 70]],
    "yellow": [[35, 255, 255], [25, 50, 70]],
    "purple": [[158, 255, 255], [129, 50, 70]],
    "orange": [[24, 255, 255], [10, 50, 70]],
    "gray": [[180, 18, 230], [0, 0, 40]],
}


def check_for_in_range(data, low, high):
    for i in range(3):
        if (data[i] > low[i] / 255.0) and (data[i] < high[i] / 255.0):
            continue
        else:
            return False
    return True


base_dir = DATA_PATH / "training"
rgb_file = base_dir / "rgb" / f"{image}.png"
depth_file = base_dir / "depth" / f"{image}_depth.png"
rgb_img = cv2.cvtColor(cv2.imread(str(rgb_file)), cv2.COLOR_BGR2RGB)
depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
cropped_depth_img = depth_img[600:1900, 1300:2700]
with open(base_dir / "ground_truth_train.json", "r") as f:
    gt_data = json.load(f)
# Load the depth image (replace 'depth.png' with your depth image)
depth = cropped_depth_img

# init a point cloud creator with the camera configuration
point_cloud_creator = PointCloudCreator(conf_file=base_dir / "oak-d-s2-poe_conf.json", logger_level=100)

files = [x.split(".")[0] for x in os.listdir(base_dir / "rgb") if ".png" in x]
for image in files:
    rgb_file = base_dir / "rgb" / f"{image}.png"
    depth_file = base_dir / "depth" / f"{image}_depth.png"
    # create pcd object
    pcd_object = point_cloud_creator.convert_depth_to_pcd(rgb_file=rgb_file, depth_file=depth_file)

    # Extract points and colors
    points = np.asarray(pcd_object.points)
    colors = np.asarray(pcd_object.colors)

    # use -y because the pixel coordinates differs from the matrix coordinate system
    # use -z because the camera is facing downwards
    x = points[:, 0]
    y = -points[:, 1]
    z = -points[:, 2]
    c = colors[:, :]

    # print(np.max(z))
    # print(np.min(z))
    # print(-0.31 - np.max(z))

    possible_ground_level = -1.04
    # # print(f"Calculated height: {-(np.max(z) - possible_ground_level)} m")

    height_limit = -0.75
    ground_level = -1.04

    # # filter out the points that are below the ground level
    mask = z > ground_level
    # # mask = z < height_limit

    # # points_filtered = points[mask]
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    c_filtered = c[mask, :]

    c_filtered_hsv = rgb_to_hsv(c_filtered)
    c_mask = [check_for_in_range(x, color_dict_HSV["green"][1], color_dict_HSV["green"][0]) for x in list(c_filtered)]
    z_filtered = z_filtered[c_mask]
    # c_mask = cv2.inRange(c_filtered_hsv, color_dict_HSV["green"][1], color_dict_HSV["green"][0])

    # print(np.min(z_filtered))
    # print(np.max(z_filtered))

    calculated_heigth = np.max(z_filtered) - np.min(z_filtered)  # + substrate_level[image[0]] / 100
    print(
        f"plant height: {calculated_heigth} m vs. gt heigth: {gt_data[image]['height']} - diff: {calculated_heigth - gt_data[image]['height']/100}"
    )
