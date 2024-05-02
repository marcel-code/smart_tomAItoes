# %%
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy.spatial import distance
from tqdm import tqdm

from src.settings import DATA_PATH
from src.utils.depth import PointCloudCreator

n = 50
max_dist = 0.05
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
    "green": [[89, 255, 255], [10, 50, 70]],
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

# %%
with open(base_dir / "ground_truth_train.json", "r") as f:
    gt_data = json.load(f)

# init a point cloud creator with the camera configuration
point_cloud_creator = PointCloudCreator(conf_file=base_dir / "oak-d-s2-poe_conf.json", logger_level=100)


# %%
res = {}
files = [x.split(".")[0] for x in os.listdir(base_dir / "rgb") if ".png" in x]
for num_neighbors in [50, 100, 150, 200, 20, 70]:
    res[num_neighbors] = {}
    for max_dist in [0.05, 0.02, 0.03, 0.01, 0.1, 0.07]:
        error = []
        error_wo = []
        res[num_neighbors][max_dist] = {}
        print(f"Starting experiment with following config: num_neighbors={num_neighbors}, max_dist={max_dist}")
        for image in tqdm(files, desc=f"Running evaluation on num_neighbors={num_neighbors} and max_dist={max_dist}"):
            try:
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

                possible_ground_level = -1.04

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
                c_mask = [
                    check_for_in_range(x, color_dict_HSV["green"][1], color_dict_HSV["green"][0])
                    for x in list(c_filtered)
                ]
                z_filtered = z_filtered[c_mask]
                x_filtered = x_filtered[c_mask]
                y_filtered = y_filtered[c_mask]

                calculated_heigth_wo = (np.max(z_filtered) - ground_level) * 100  # + substrate_level[image[0]] / 100

                # %%
                img = np.stack([x_filtered, y_filtered, z_filtered], axis=-1)
                found = False

                D = distance.squareform(distance.pdist(img))
                i = 0
                max_idx = np.argsort(img[:, 2])[::-1]
                closest = np.sort(D, axis=1)
                while not found and i < img.shape[0] * 0.05:
                    # number of elements closer than x
                    # print(closest)
                    num_elem = np.sum(closest[max_idx[i]] < max_dist)
                    # print(num_elem)
                    # if number > n than found = true, else set z_filtered to ground truth
                    if num_elem > num_neighbors:
                        found = True
                    else:
                        img[max_idx[i], 2] = ground_level
                    i = i + 1
                x_filtered = img[:, 0]
                y_filtered = img[:, 1]
                z_filtered = img[:, 2]

                calculated_heigth = (np.max(z_filtered) - ground_level) * 100  # + substrate_level[image[0]] / 100

                res[num_neighbors][max_dist][image] = {
                    "calculated_wo": calculated_heigth_wo,
                    "calculated_w": calculated_heigth,
                    "gt": gt_data[image]["height"],
                    "error_wo": (calculated_heigth_wo - gt_data[image]["height"]) ** 2,
                    "error_w": (calculated_heigth - gt_data[image]["height"]) ** 2,
                }
                error.append((calculated_heigth - gt_data[image]["height"]) ** 2)
                error_wo.append((calculated_heigth_wo - gt_data[image]["height"]) ** 2)
            except Exception as e:
                print(f"Error: {e}. Next iteration startet.")
        res[num_neighbors][max_dist]["error_w"] = np.sum(error) / len(error)
        res[num_neighbors][max_dist]["error_wo"] = np.sum(error_wo) / len(error)

        with open(
            f"outputs/analysis/error_calc_num_nn_{num_neighbors}_max_dist_{str(max_dist).replace('.','_')}.json",
            "w",
        ) as f:
            json.dump(res[num_neighbors][max_dist], f)

# save overall file
with open("outputs/analysis/error_calc.json", "w") as f:
    json.dump(res, f)
