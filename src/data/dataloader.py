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
from ..utils.depth import PointCloud
from ..utils.image import numpy_image_to_torch, read_image
from ..utils.tools import fork_rng
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class TomatoDataset(BaseDataset):
    default_conf = {
        # image search
        "data_dir": "training",  # the top-level directory
        "image_dir": "rgb/",  # the subdirectory with the images
        "depth_dir": None,
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
        "num_workers": 2,
    }

    def _init(self, conf):
        data_dir = DATA_PATH / conf.data_dir

        self.pcloud = PointCloud(DATA_PATH / conf.data_dir / "oak-d-s2-poe_conf.json")

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
        self.grayscale = conf.grayscale
        self.resize = conf.resize
        self.ground_truth = ground_truth
        self.pcloud = pcloud

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

    def _preprocess_depth_data(self, img):
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
        pointcloud = self.pcloud.convert_depth_to_point_array(depth_img=img)
        depth = torch.asarray(max(pointcloud[:, 2]), dtype=torch.float32).reshape((1))
        return depth

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
        depth = self._preprocess_depth_data(img_depth)

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
