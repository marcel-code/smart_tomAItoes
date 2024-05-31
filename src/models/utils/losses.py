import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

src_pth = Path(__file__).parent.parent.parent.absolute()  # somehow not working for input
sys.path.append("..\\..\\..")  # using relativ import - seems to work
# sys.path.append("C:\\Users\\Marcel\\OneDrive - student.kit.edu\\Dokumente\\projects\\Projects\\smart_tomAItoes")
# from settings import DATA_PATH
from src.settings import DATA_PATH


class DummyLoss(torch.nn.Module):
    def __init__(self):
        super(DummyLoss, self).__init__()

    def forward(self, inputs, targets):
        loss = -1 * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        return loss.mean()


class ProvidedLoss(torch.nn.Module):
    def __init__(self):
        super(ProvidedLoss, self).__init__()

    def forward(self, pred: dict, truth: dict) -> float:
        # TODO Implementation error handling if needed
        error = 0
        for trait in ["height", "fw_plant", "leaf_area", "number_of_red_fruits"]:
            diff = [((pred[i][trait] - truth[i][trait]) / (truth[i][trait] + 1)) ** 2 for i in truth]
            error += torch.sqrt(torch.nanmean(torch.Tensor(diff)))
        return error / 4


class ProvidedLossTraining(torch.nn.Module):
    def __init__(self):
        super(ProvidedLossTraining, self).__init__()

    def forward(self, pred: dict, truth: dict) -> float:
        # TODO Implementation error handling if needed
        # TODO Handling nans!
        traits = pred.size(1)
        error = 0
        for trait in range(traits - 1):  # Iterate over traits
            diff = ((pred[:, trait] - truth[:, trait]) / (truth[:, trait] + 1)) ** 2
            error += torch.sqrt(torch.nanmean(diff))

        return error / 4


if __name__ == "__main__":
    # NOTE: For check, you need to be in the path of this file!
    # define filenames
    dummy_pred_filename = "demo_submission_train.json"
    dummy_gt_filename = "ground_truth_train.json"
    gt_error_filename = "demo_response_train.txt"

    # get file paths
    dummy_pred_path = os.path.join(DATA_PATH, "training", dummy_pred_filename)
    dummy_gt_path = os.path.join(DATA_PATH, "training", dummy_gt_filename)
    gt_error_path = os.path.join(DATA_PATH, "training", gt_error_filename)

    # get file data
    with open(dummy_pred_path) as f:
        pred = json.load(f)
    with open(dummy_gt_path) as g:
        truth = json.load(g)
    with open(gt_error_path, "r") as f:
        gt_error_check = float(f.readline())

    assert len(pred.keys()) == len(truth.keys())
    print(f"Len of keys identical = {len(pred.keys())} and {len(truth.keys())}")
    # get error caluclator
    loss_fn = ProvidedLoss()
    loss = loss_fn(pred, truth)
    # check target error and calculated error
    print(f"Error calculated: {loss} - target error: {gt_error_check} - difference: {loss - gt_error_check}")
    assert loss == gt_error_check
    print("Target and calulated loss similar! Check successfull!")
