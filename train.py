import argparse
import copy
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from src import logger
from src.data.dataloader import TomatoDataset
from src.models import get_model
from src.settings import TRAINING_PATH
from src.utils.misc import get_ground_truth_dict, get_ground_truth_tensor, get_output_dict

## TODO SECTION

# TODO Setting up evaluation pipeline
# TODO moving get_* function to utils
# TODO implementation of evaluation
# TODO Dataloader depth image inclusion
# TODO Modell nach training abspeichern
# TODO fix issue with batchsize = num:workers in yaml
# TODO implementation of colab run - therefore: inclusion of all necessary packages, ...
# TODO undo deletion of scaling of image in numpy to torch (load_image)
# TODO only store non pre trained weights when using vgg or similar
# TODO restoring training and fine-tune or prolong training
# TODO Check if backbone is trained aswell

# TODO LIST
# TODO DAtaloader adaption
# TODO Model adaption

default_train_conf = {
    "seed": "???",  # training seed
    "epochs": 1,  # number of epochs
    "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
    "opt_regexp": None,  # regular expression to filter parameters to optimize
    "optimizer_options": {},  # optional arguments passed to the optimizer
    "lr": 0.001,  # learning rate
    "lr_schedule": {
        "type": None,  # string in {factor, exp, member of torch.optim.lr_scheduler}
        "start": 0,
        "exp_div_10": 0,
        "on_epoch": False,
        "factor": 1.0,
        "options": {},  # add lr_scheduler arguments here
    },
    "lr_scaling": [(100, ["dampingnet.const"])],
    "eval_every_iter": 1000,  # interval for evaluation on the validation set
    "save_every_iter": 5000,  # interval for saving the current checkpoint
    "log_every_iter": 200,  # interval for logging the loss to the console
    "log_grad_every_iter": None,  # interval for logging gradient hists
    "test_every_epoch": 1,  # interval for evaluation on the test benchmarks
    "keep_last_checkpoints": 10,  # keep only the last X checkpoints
    "load_experiment": None,  # initialize the model from a previous experiment
    "median_metrics": [],  # add the median of some metrics
    "recall_metrics": {},  # add the recall of some metrics
    "pr_metrics": {},  # add pr curves, set labels/predictions/mask keys
    "best_key": "loss/total",  # key to use to select the best checkpoint
    "dataset_callback_fn": None,  # data func called at the start of each epoch
    "dataset_callback_on_val": False,  # call data func on val data?
    "clip_grad": None,
    "pr_curves": {},
    "plot": None,
    "submodules": [],
}


def filter_parameters(params, regexp):
    """Filter trainable parameters based on regular expressions."""

    # Examples of regexp:
    #     '.*(weight|bias)$'
    #     'cnn\.(enc0|enc1).*bias'
    def filter_fn(x):
        n, p = x
        match = re.search(regexp, n)
        if not match:
            p.requires_grad = False
        return match

    params = list(filter(filter_fn, params))
    assert len(params) > 0, regexp
    logger.info("Selected parameters:\n" + "\n".join(n for n, p in params))
    return params


def pack_lr_parameters(params, base_lr, lr_scaling):
    """Pack each group of parameters with the respective scaled learning rate."""
    filters, scales = tuple(zip(*[(n, s) for s, names in lr_scaling for n in names]))
    scale2params = defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logger.info(
        "Parameters with scaled learning rate:\n%s",
        {s: [n for n, _ in ps] for s, ps in scale2params.items() if s != 1},
    )
    lr_params = [{"lr": scale * base_lr, "params": [p for _, p in ps]} for scale, ps in scale2params.items()]
    return lr_params


def get_lr_scheduler(optimizer, conf):
    """Get lr scheduler specified by conf.train.lr_schedule."""
    if conf.type not in ["factor", "exp", None]:
        return getattr(torch.optim.lr_scheduler, conf.type)(optimizer, **conf.options)

    # backward compatibility
    def lr_fn(it):  # noqa: E306
        if conf.type is None:
            return 1
        if conf.type == "factor":
            return 1.0 if it < conf.start else conf.factor
        if conf.type == "exp":
            gam = 10 ** (-1 / conf.exp_div_10)
            return 1.0 if it < conf.start else gam
        else:
            raise ValueError(conf.type)

    return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)


# from torch.utils.tensorboard import SummaryWriter
def train_one_epoch(epoch_index, training_loader, optimizer, model, num_batches=5):
    running_loss = 0.0
    last_loss = 0.0
    model.train(True)
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(training_loader), desc="Model Training"):
        # Every data instance is an input + label pair
        inputs = data["rgb"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # post-process data for loss calc
        ground_truth = get_ground_truth_tensor(data)

        # Compute the loss and its gradients
        loss = model.loss(outputs, ground_truth)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % num_batches == (num_batches - 1):
            last_loss = running_loss / num_batches  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            # tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def train(conf):
    data_conf = copy.deepcopy(conf.data)
    print(data_conf)

    data_loader = TomatoDataset(data_conf)

    train_loader = data_loader.get_data_loader("train")
    val_loader = data_loader.get_data_loader("val")
    # test_loader = data_loader.get_data_loader("test")
    # TODO fix collate issue -> enabling loading of all datatypes

    # Dynamic loading of model (need to be defined in the model section (single .py for a model class))
    model = get_model(conf.model.name)(conf)

    # print model summary (layer, parameter, ...)
    summary(model, tuple(conf.model.input_shape))

    optimizer_fn = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }[conf.train.optimizer.name]
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # if conf.train.opt_regexp:
    #     params = filter_parameters(params, conf.train.opt_regexp)
    all_params = [p for n, p in params]

    lr_params = pack_lr_parameters(
        params, conf.train.optimizer.lr, [(100, ["dampingnet.const"])]
    )  # conf.train.lr_scaling)
    optimizer = optimizer_fn(lr_params, lr=conf.train.optimizer.lr, **conf.train.optimizer.optimizer_options)
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, conf=conf.train.lr_schedule)

    # optimizer = torch.optim.SGD(model.parameters(), lr=conf.train.optimizer.lr, momentum=conf.train.optimizer.momentum)
    # lr_params = pack_lr_parameters(params, conf.train.lr, conf.train.lr_scaling)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("outputs/training/{}/tensorboard_{}".format(conf.experiment, timestamp))

    epoch_number = 0
    best_vloss = 10000000

    # Optimizers specified in the torch.optim package

    while epoch_number < conf.train.epochs:
        print("EPOCH {}:".format(epoch_number + 1))

        # update learning rate
        if conf.train.lr_schedule.on_epoch and epoch_number > 0:
            old_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step()
            logger.info(f'lr changed from {old_lr} to {optimizer.param_groups[0]["lr"]}')

        avg_loss = train_one_epoch(epoch_number, train_loader, optimizer, model)

        # evaluate model
        running_vloss = 0.0
        model.eval()

        # TODO include evaluation section
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata["rgb"], get_ground_truth_tensor(vdata)
                voutputs = model(vinputs)
                vloss = model.loss(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        writer.add_scalar("train/loss", avg_loss, epoch_number + 1)
        writer.add_scalar("train/lr", np.array(lr_scheduler.get_last_lr()), epoch_number + 1)
        writer.add_scalar("train/epoch", epoch_number + 1, epoch_number + 1)
        writer.add_scalar("val/loss", avg_loss, epoch_number + 1)
        writer.add_scalar("val/epoch", epoch_number + 1, epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = Path(
                "outputs/training/{}/{}_model_{}_epoch_{}".format(
                    conf.experiment, timestamp, conf.model.name, epoch_number
                )
            )
            print(f"New best model saved at epoch {epoch_number}")
            torch.save(model.state_dict(), model_path)

        # TODO Implementation of evaluation procedure
        # TODO Inclusion of additional metric to tensorboard and overview
        epoch_number = epoch_number + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, default="Test")
    parser.add_argument("--conf", type=str, default=".\\configs\\train.yaml")
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
    output_dir = Path(TRAINING_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(args)

    if args.conf:
        conf = OmegaConf.load(args.conf)

    if conf.train.seed is None:
        conf.train.seed = torch.initial_seed() & (2**32 - 1)
    OmegaConf.save(conf, str(output_dir / "config.yaml"))
    conf.experiment = args.experiment

    train(conf)
