"""
Simply load images from a folder or nested folders (does not have any split),
and apply homographic adaptations to it. Yields an image pair without border
artifacts.
"""

import argparse
import json
import logging
import os
import shutil
import tarfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..settings import DATA_PATH
from ..utils.image import numpy_image_to_torch, read_image
from .base_dataset import BaseDataset
from .pointcloud_tools import PointCloudCreator

logger = logging.getLogger(__name__)

POT_HEIGHT = {
    "A": 0.1,
    "B": 0.07,
    "D": 0.1,
    "C": 0.1,
}
SUBSTRATE_HIGH = {  # if base ground (orange) is measured at -1.15
    "A": 0.0,
    "B": 0.075,
    "C": 0.025,
    "D": 0.0,
}

SUBSTRATE_LOW = {  # if base ground (orange) is measured at -1.35
    "A": 0.0,
    "B": 0.065,
    "C": 0.013,
    "D": 0.0,
}
GROUND_LEVEL_MIN = -1.4
GROUND_LEVEL_MAX = -1.1
MIN_FREQUENCY_PCLOUD_POINTS = 20
NUM_BINS = 300
EST_GROUND_LEVEL = 1.148


class TomatoDataset(BaseDataset):
    default_conf = {
        # image search
        "data_dir": "training",  # the top-level directory
        "image_dir": "rgb/",  # the subdirectory with the images
        "depth_dir": None,
        "depth_prep_dir": "depth_prep_dir",
        "image_list": "revisitop1m.txt",  # optional: list or filename of list
        "input_shape": [256, 256, 3],
        "resize": [340, 240],
        "glob": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        # splits
        "train_size": 100,
        "val_size": 10,
        "test": False,
        "shuffle_seed": 0,  # or None to skip
        # image loading
        "grayscale": False,
        "reseed": False,
        "num_workers": 1,
    }

    def _init(self, conf):
        data_dir = DATA_PATH / conf.data_dir

        self.pcloud = DATA_PATH / conf.data_dir / "oak-d-s2-poe_conf.json"

        if not data_dir.exists():
            raise FileNotFoundError(data_dir)
        else:
            print(f"Data for Training found at {data_dir}")

        image_dir = data_dir / conf.image_dir
        images = []
        for i in os.listdir(image_dir):
            images.append(i)

        if conf.shuffle_seed is not None:
            np.random.RandomState(conf.shuffle_seed).shuffle(images)
        if not conf.test:
            train_images = images[: conf.train_size]
            val_images = images[conf.train_size : conf.train_size + conf.val_size]
            self.images = {"train": train_images, "val": val_images}

            # read ground truth data
            ground_truth_train = "ground_truth_train.json"

            # get file data
            with open(os.path.join(DATA_PATH, "training", ground_truth_train)) as g:
                self.ground_truth = json.load(g)
        else:
            self.images = {"test": images}
            self.ground_truth = {}

    def get_dataset(self, split):
        return _Dataset(self.conf, self.images[split], split, self.ground_truth, self.pcloud)


class _Dataset(torch.utils.data.Dataset):
    dataset = 0

    def __init__(self, conf, image_names, split, ground_truth, pcloud):
        self.conf = conf
        self.split = split
        self.image_names = np.array(image_names)
        self.image_dir = DATA_PATH / conf.data_dir / conf.image_dir
        self.depth_dir = DATA_PATH / conf.data_dir / conf.depth_dir
        self.depth_prep_dir = DATA_PATH / conf.data_dir / conf.depth_prep_dir
        self.grayscale = conf.grayscale
        self.resize = conf.resize
        self.pcloud = pcloud
        self.ground_truth = ground_truth

        # Either create preprocessed depth folder or read all depth data file names
        if not os.path.isdir(self.depth_prep_dir):
            os.mkdir(self.depth_prep_dir)
            print("Preprocessed depth folder created.")

        _Dataset.dataset = _Dataset.dataset + 1

        print(f"Dataset {_Dataset.dataset} created")

    def __getitem__(self, idx):
        return self.getitem(idx)

    def _preprocess_rgb_data(self, img):
        """Preprocess rgb data including image to torch conversion

        Parameters
        ----------
        img : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
        return numpy_image_to_torch(img)

    def _preprocess_depth_data(self, img, name, depth=torch.tensor([0])):
        """Preprocess Depth data including

        Parameters
        ----------
        img : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        img_path = str(self.image_dir / f"{name}.png")
        depth_path = str(self.depth_dir / f"{name}_depth.png")

        pcloud = PointCloudCreator(self.pcloud, logger_level=100)
        pcloud_arr = pcloud.convert_depth_to_pcd(img_path, depth_path)

        # Extract points and colors
        points = np.asarray(pcloud_arr.points)
        z = -points[range(0, len(points), 100), 2]
        z_ground_filtered = z[z > GROUND_LEVEL_MIN]

        # Extract plant top level
        freq, bins = np.histogram(z_ground_filtered, bins=NUM_BINS)

        mask = freq > MIN_FREQUENCY_PCLOUD_POINTS
        freq = freq[mask]
        bins = bins[:-1][mask]

        plant_top = np.max(bins)

        # Make sure we only have real ground points
        z_only_ground = z_ground_filtered[z_ground_filtered < GROUND_LEVEL_MAX]

        # Extract ground level
        freq_2, bins_2 = np.histogram(z_only_ground, bins=NUM_BINS)

        ground_bin = np.argmax(freq_2)  # indize of the most frequent value
        ground_value = bins_2[ground_bin]  # most frequent value

        if ground_value < -1.25:
            substrat = SUBSTRATE_LOW[name[0]]
        else:
            substrat = SUBSTRATE_HIGH[name[0]]

        # Calculate plant height
        plant_height = plant_top - ground_value - POT_HEIGHT[name[0]] - substrat

        print(f"Ground level idx: {np.argmax(freq_2)}")
        print(f"Ground level value: {bins_2[ground_bin]}")
        print(f"Plant top: {np.max(bins)}")
        print(f"Plant height: {plant_height}")
        with open(f"{self.depth_prep_dir}/{name}.txt", "w") as f:
            f.write(str(plant_height))
        print(f"Preprocessing for depth of {name} stored in according file.")
        plant_height = torch.tensor(plant_height)
        plant_height = plant_height.reshape((1, -1))
        return plant_height

    def getitem(self, idx):
        name = self.image_names[idx].split(".")[0]

        # Read all images
        # TODO read_image to load_image + grayscale inclusion
        img_rgb = read_image(self.image_dir / f"{name}.png", self.grayscale)
        img_depth = cv2.imread(str(self.depth_dir / f"{name}_depth.png"), cv2.IMREAD_UNCHANGED)
        # img_irL = read_image(self.depth_dir / f"{name}_irL.png")
        # img_irR = read_image(self.depth_dir / f"{name}_irR.png")

        if img_rgb is None:
            raise FileNotFoundError(f"Image {name} not known")
        size = img_rgb.shape[:2][::-1]

        img_rgb = self._preprocess_rgb_data(img_rgb)

        if f"{name}.txt" in os.listdir(self.depth_prep_dir):
            with open(f"{self.depth_prep_dir}/{name}.txt", "rb") as f:
                depth = float(f.readline())
        else:
            depth = self._preprocess_depth_data(img_depth, name)

        depth = torch.tensor([depth], dtype=torch.float32)

        # TODO read target values
        data = {
            "name": name,
            "rgb": img_rgb,
            "depth": depth,
            # "irL": img_irL,
            # "irR": img_irR,
            # "idx": idx,
            # "category": name[0],
            "label": np.random.randint(2, size=2) if self.split in ["train", "val"] else [],
            "ground_truth": self.ground_truth[name] if self.split in ["train", "val"] else {},
            "pred": {},
        }

        return data

    def __len__(self):
        return len(self.image_names)


if __name__ == "__main__":
    raise NotImplementedError("Main in dataloader not implemented")
