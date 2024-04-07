import argparse
import copy
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import logger
from src.data.dataloader import TomatoDataset
from src.models import get_model
from src.settings import TRAINING_PATH


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
        labels = data["label"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = model.loss(outputs, labels)
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

    # TODO fix collate issue -> enabling loading of all datatypes

    # Dynamic loading of model (need to be defined in the model section (single .py for a model class))
    model = get_model(conf.model.name)(conf)
    # model = DummyModel(conf)
    optimizer = torch.optim.SGD(model.parameters(), lr=conf.train.optimizer.lr, momentum=conf.train.optimizer.momentum)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("outputs/training/{}/tensorboard_{}".format(conf.experiment, timestamp))

    epoch_number = 0
    best_vloss = 10000000

    # Optimizers specified in the torch.optim package

    while epoch_number < conf.train.epochs:
        print("EPOCH {}:".format(epoch_number + 1))

        avg_loss = train_one_epoch(epoch_number, train_loader, optimizer, model)

        # evaluate model
        running_vloss = 0.0
        model.eval()

        # TODO include evaluation section
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata["rgb"], vdata["label"]
                voutputs = model(vinputs)
                vloss = model.loss(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss", {"Training": avg_loss, "Validation": avg_vloss}, epoch_number + 1
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = Path(
                "outputs/training/{}/{}_model_{}_epoch_{}".format(
                    conf.experiment, timestamp, conf.model.name, epoch_number
                )
            )
            torch.save(model.state_dict(), model_path)

        # TODO Generation of simple model for training -> loading and training function
        # TODO implementation of loss function

        # TODO Implementation of evaluation procedure
        epoch_number = epoch_number + 1


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


# TODO LIST
# TODO DAtaloader adaption
# TODO Model adaption
# TODO Evaluation scipt
# TODO Concept creation
