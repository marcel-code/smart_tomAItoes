import json
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np

# TODO import logger
# TODO include dummy image


class PointCloud:
    def __init__(self, path: Path, depth_scale: float = 0.001, depth_trunc: int = 20000):
        camera_data = json.load(open(path), "r")

        self.intrinsics = camera_data["color_int"]
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc

        # init a dummy placeholder to speed up the generation of images
        self.dummy_x = self.intrinsics["height"]  # TODO Check for correctness
        self.dummy_y = self.intrinsics["width"]  # TODO Check for correctness
        self.dummy_img = np.array(list(product(np.arange(0, self.dummy_y), np.arange(0, self.dummy_x))))
        self.dummy_index = np.zeros(self.dummy_x * self.dummy_y, np.float64)

    def convert_depth_to_point_array(self, depth_file: Path) -> np.array:
        """This is a custom version to reconstruct the point cloud. It is a bit slower but could useful
        if you want the point cloud to be exactly the same size as the image

        Parameters
        ----------
        depth_file: Path
            file to the depth image

        Returns
        -------
        points_array: np.array
            An array with x,y,z coordinates of each pixel.

        """
        t0 = time.time()
        # TODO include logger
        # self.logger.info(f"Converting depth image file {depth_file} to points array.")

        intrinsics = self.intrinsics["intrinsics"]
        depth_scale = self.depth_scale

        depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)

        # copy indexes
        y_index = self.dummy_img[:, 0].copy()
        x_index = self.dummy_img[:, 1].copy()

        # convert complete array no forloop needed. X en Y are vectors!
        Z = depth_scale * depth_img[y_index, x_index]
        X = self.dummy_index.copy()
        Y = self.dummy_index.copy()

        z_bool = Z != 0

        X[z_bool] = Z[z_bool] * (x_index[z_bool] - intrinsics["ppx"]) / intrinsics["fx"]
        Y[z_bool] = Z[z_bool] * (y_index[z_bool] - intrinsics["ppy"]) / intrinsics["fy"]

        points_array = np.array([X, Y, Z], dtype=np.float64).transpose()
        # self.logger.info(
        #     f"Generated points array of {len(points_array)=} for {depth_file} in {time.time() - t0:.2f} seconds."
        # )

        return points_array
