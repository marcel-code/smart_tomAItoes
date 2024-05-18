import json
import os

import numpy as np

from src.settings import DATA_PATH

gt_file = DATA_PATH / "training" / "ground_truth_train.json"
with open(gt_file, "rb") as f:
    gt = json.load(f)

prediction = {}
for file in os.listdir(DATA_PATH / "training" / "depth_prep_dir"):
    with open(DATA_PATH / "training" / "depth_prep_dir" / file, "rb") as f:
        prediction[file.split(".")[0]] = float(f.readline())

print(prediction)
print(gt)

error_dict = {
    "A": [],
    "B": [],
    "C": [],
    "D": [],
}


for k, v in prediction.items():
    error_dict[k[0]].append((v - gt[k]["height"]) ** 2)

for k, v in error_dict.items():
    error_dict[k] = np.sum(error_dict[k]) / len(error_dict[k])

print(error_dict)
