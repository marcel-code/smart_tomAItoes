import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src import logger
from src.data.dataloader import TomatoDataset
from src.models.initialModel import initialModel
from src.settings import EVAL_PATH

## TODO SECTION
# TODO Dataloader testing test


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


def get_output_dict(
    pred: torch.Tensor, data: dict, key_list=["height", "fw_plant", "leaf_area", "number_of_red_fruits"]
):
    """Conversion of model output () to dict"""
    # TODO Check for correctness
    res = {}
    for i in range(len(data["name"])):
        res[data["name"][i]] = {}
        for j in range(len(key_list)):
            res[data["name"][i]][key_list[j]] = pred[i][j].item()
    return res


# @torch.no_grad
def evaluation(conf):
    data_conf = copy.deepcopy(conf.data)
    print(data_conf)
    validation_dir = os.path.join(EVAL_PATH, conf.experiment)

    data_loader = TomatoDataset(data_conf)

    test_loader = data_loader.get_data_loader("test")

    model = initialModel(conf)
    model.load_model(conf.model.pretrained_model)
    model.eval()

    pred_dict = {}
    cnt = 0
    for i, data in tqdm(enumerate(test_loader), desc="Evaluation"):
        inputs = data["rgb"]  # currently only gray values
        pred = model(inputs)
        # print(np.array(model(inputs)[0]))
        # pred = model(xy) - Take model 20240407_030148_model_DummyModel_epoch_0 (should hopefully work). You need to
        # load the model class first (DummyModel in this case) and then add the weights stored in the mentioned file

        output_dict = get_output_dict(pred, data)
        print(output_dict)

        # Append output_dict to pred_dict
        for key, value in output_dict.items():
            if key in pred_dict:
                pred_dict[key].extend(value)
            else:
                pred_dict[key] = value

    # Write the results to the submission file
    submission_file = os.path.join(EVAL_PATH, conf.experiment, "validation_submission.json")
    os.makedirs(os.path.dirname(submission_file), exist_ok=True)
    with open(submission_file, "w") as f:
        json.dump(pred_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, default="Test")
    parser.add_argument("--conf", type=str, default=".\\configs\\test.yaml")
    parser.add_argument(
        "--mixed_precision",
        "--mp",
        default=None,
        type=str,
        choices=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--compile",
        default=None,
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--print_arch", "--pa", action="store_true")
    parser.add_argument("--detect_anomaly", "--da", action="store_true")
    parser.add_argument("--log_it", "--log_it", action="store_true")
    parser.add_argument("--no_eval_0", action="store_true")
    parser.add_argument("--run_benchmarks", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    logger.info(f"Starting experiment {args.experiment}")
    output_dir = Path(EVAL_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(args)

    if args.conf:
        conf = OmegaConf.load(args.conf)

    if conf.train.seed is None:
        conf.train.seed = torch.initial_seed() & (2**32 - 1)
    OmegaConf.save(conf, str(output_dir / "config.yaml"))
    conf.experiment = args.experiment

    evaluation(conf)
