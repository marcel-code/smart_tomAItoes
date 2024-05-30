import json
import os

from src.models.utils.losses import ProvidedLoss
from src.settings import DATA_PATH, EVAL_PATH

folder = "validation"
experiment = "train_w_gt"

# get file paths
pred_path = os.path.join(EVAL_PATH, experiment, "validation_submission.json")
gt_path = os.path.join(DATA_PATH, folder, "ground_truth_validation.json")

# get file data
with open(pred_path) as f:
    pred = json.load(f)
with open(gt_path) as g:
    truth = json.load(g)

assert len(pred.keys()) == len(truth.keys())
print(f"Len of keys identical = {len(pred.keys())} and {len(truth.keys())}")
# get error caluclator
loss_fn = ProvidedLoss()
loss = loss_fn(pred, truth)
# check target error and calculated error
print(f"Error calculated: {loss}")
