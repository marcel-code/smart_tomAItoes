import numpy as np
import torch
from matplotlib.colors import rgb_to_hsv

from point_cloud_demo.scripts.pointcloud_tools import PointCloudCreator


def get_ground_truth_tensor(data, key_list=["height", "fw_plant", "leaf_area", "number_of_red_fruits"]):
    """Conversion of dict to tensor for loss calculation"""
    res = torch.Tensor(len(data["name"]), len(key_list))
    for i in range(len(data["name"])):
        for j in range(len(key_list)):
            res[i, j] = data["ground_truth"][key_list[j]][i]

    return res


def get_ground_truth_dict(data):
    return {
        data["name"][i]: {
            "height": data["ground_truth"]["height"][i],
            "fw_plant": data["ground_truth"]["fw_plant"][i],
            "leaf_area": data["ground_truth"]["leaf_area"][i],
            "number_of_red_fruits": data["ground_truth"]["number_of_red_fruits"][i],
        }
        for i in range(len(data["name"]))
    }


def get_output_dict(pred, data, key_list=["height", "fw_plant", "leaf_area", "number_of_red_fruits"]):
    """Conversion of model output () to dict"""
    # TODO Check for correctness
    res = {}
    for i in range(len(data["name"])):
        res[data["name"][i]] = {}
        for j in range(len(key_list)):
            res[data["name"][i]][key_list[j]] = pred[i][j]
    return res


SUBSTRATE_LEVEL = {
    "A": 10,
    "B": 7,
    "C": 10,
    "D": 10,
}

COLOR_DICT_HSV = {
    "black": [[180, 255, 30], [0, 0, 0]],
    "white": [[180, 18, 255], [0, 0, 231]],
    "red1": [[180, 255, 255], [159, 50, 70]],
    "red2": [[9, 255, 255], [0, 50, 70]],
    "green": [[89, 255, 255], [50, 50, 70]],
    "blue": [[128, 255, 255], [90, 50, 70]],
    "yellow": [[35, 255, 255], [25, 50, 70]],
    "purple": [[158, 255, 255], [129, 50, 70]],
    "orange": [[24, 255, 255], [16, 185, 185]],
    "gray": [[180, 18, 230], [0, 0, 40]],
}

MAX_GROUND_HEIGHT = -1
MIN_GROUND_HEIGHT = -1.5
MAX_PLANT_HEIGHT = -0.75
X_BOARDER = 0.5
Y_BOARDER = 0.5

steps = 410


def check_for_in_range(data, low, high):
    """
    Function to check whether hsv value is in range or not.
    """
    for i in range(3):
        if (data[i] > (low[i] / 255.0)) and (data[i] < (high[i] / 255.0)):
            continue
        else:
            return False
    return True


def compare_list_element(list1, list2):
    """
    Function to compare two boolean arrays.
    Aim: finding the indices where only the first array is true and the second false.
    """
    result = [list1[idx] and not list2[idx] for idx in range(len(list1))]
    return result


def ground_level_detection(
    pcd_object: PointCloudCreator,
) -> float:
    """
    Function to detect ground level of plant
    """

    # Extract points and colors
    points = np.asarray(pcd_object.points)
    colors = np.asarray(pcd_object.colors)

    # use -y because the pixel coordinates differs from the matrix coordinate system
    # use -z because the camera is facing downwards
    x = points[::3, 0]
    y = -points[::3, 1]
    z = -points[::3, 2]
    c = colors[::3, :]

    # # filter out the points that are below the ground level
    mask = (
        (z > MIN_GROUND_HEIGHT)
        & (z < MAX_PLANT_HEIGHT)
        & (x > -X_BOARDER)
        & (x < X_BOARDER)
        & (y > -Y_BOARDER)
        & (y < Y_BOARDER)
    )

    # # points_filtered = points[mask]
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    c_filtered = c[mask, :]

    c_filtered_hsv = rgb_to_hsv(c_filtered)
    # Plant segmentation
    c_mask_plant = [
        check_for_in_range(x, COLOR_DICT_HSV["green"][1], COLOR_DICT_HSV["green"][0]) for x in list(c_filtered)
    ]
    # Ground segmentation
    mask = (z_filtered < MAX_GROUND_HEIGHT) & (z_filtered > MIN_GROUND_HEIGHT)
    x_filtered = x_filtered[mask]
    y_filtered = y_filtered[mask]
    z_filtered = z_filtered[mask]
    c_filtered = c_filtered[mask, :]
    c_filtered_hsv = rgb_to_hsv(c_filtered)
    c_mask_ground = [
        check_for_in_range(x, COLOR_DICT_HSV["orange"][1], COLOR_DICT_HSV["orange"][0]) for x in list(c_filtered_hsv)
    ]
    c_mask_ground = compare_list_element(c_mask_ground, c_mask_plant)

    # Potentially more robust - but probably not needed
    # x_filtered_ground = x_filtered[c_mask_ground]
    # y_filtered_ground = y_filtered[c_mask_ground]
    # z_filtered_ground = z_filtered[c_mask_ground]

    # Get histogramm of highest orange density
    bins = np.linspace(MIN_GROUND_HEIGHT, MAX_GROUND_HEIGHT, steps)
    freq, bins = np.histogram(z_filtered, bins)
    ground_level = bins[np.argmax(freq)] + (MAX_GROUND_HEIGHT - MIN_GROUND_HEIGHT) / steps * 0.5
    return ground_level
