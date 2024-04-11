import argparse
import copy
import json
import os

# from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src import logger
from src.data.dataloader import TomatoDataset
from src.models import get_model
from src.settings import EVAL_PATH

# TODO SECTION
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

    # data komt als array raus --> 1x4 array


# TODO change back to 4 parameters
# def get_output_dict(pred, data, key_list=["height", "fw_plant", "leaf_area", "number_of_red_fruits"]):
def get_output_dict(pred, data, key_list=["height", "fw_plant"]):
    """Conversion of model output () to dict"""
    # TODO Check for correctness
    res = {}
    for i in range(len(data["name"])):
        res[data["name"][i]] = {}
        for j in range(len(key_list)):
            res[data["name"][i]][key_list[j]] = pred[i][j]

    # TODO change the names of res and res_dict to something that makes more sense
    # TODO should we write another function rather than messing up this one?

    # res is a dictionary build from the tensor pred
    # meaning that the brackets of the tensor get in the way
    # if we convert the tensor pred to a list we can use the list to build the dictionary --> pred.tolist()
    # still the list will have some brackets that we dont want --> [0.0, 0.0, 0.0, 0.0] around each loop of data
    # therefore we flatten the dictionary to become a list again
    # and then convert the list back into a dictionary

    # Flatten the dictionary into a list of tuples
    flat_list = [(k1, k2, v) for k1, inner_dict in res.items() for k2, v in inner_dict.items()]

    # Convert the list of back into a dictionary
    res_dict = {}
    for k1, k2, v in flat_list:
        if k1 in res_dict:
            res_dict[k1][k2] = v
        else:
            res_dict[k1] = {k2: v}
    return res_dict

    # data --> vom data loader

    # die zwei arrays in data sind die Formate die in evaluation(conf) reinkommen
    # result = []
    # result_dict = {}


# evaluation(conf) mit meiner evaluation funktion anpassen
# model einlesen (DummyModel)
# mit den gewichten aus dem trainierten Modell kombinieren (pred = ...._...._model_DummyModel_epoch_0)
# get_data_loader holt aus dem TomatoDataset die Testdaten raus ("test")
# dann wollen wir die inputs (die Bilder aus dem Testdatensatz) pro Loop 1 Bild einladen
# der input kommt dann vermutlich ins das Modell und gibt eine Prediction aus (4 Parameter)
# die 4 Parameter werden dann in ein Dictionary umgewandelt (get_output_dict) und abgespeichert
# das gleiche wird dann für die ground_truth gemacht (ab diesem Schritt Vorschlag von Copilot für Folgeschritte)
# dann wird die loss berechnet (ProvidedLoss) und abgespeichert
# das ganze wird dann für alle Bilder im Testdatensatz gemacht
# am Ende wird dann der Durchschnitt der Loss berechnet und ausgegeben


def evaluation(conf):
    data_conf = copy.deepcopy(conf.data)
    print(data_conf)

    data_loader = TomatoDataset(data_conf)

    test_loader = data_loader.get_data_loader("test")

    # model = DummyModel(conf)

    # TODO change back to get_model function if tested
    # TODO throw out DummyModel import at top
    model = get_model(conf.model.name)(conf)

    optimizer = torch.optim.SGD(model.parameters(), lr=conf.train.optimizer.lr, momentum=conf.train.optimizer.momentum)

    # TODO discuss with Marcel wether it is enough to load the state_dict as follows without
    # loading the optimizer and model seperately. Doesnt work with the keywords that i tried

    path_to_model_checkpoint = "20240407_030148_model_DummyModel_epoch_0"
    checkpoint = torch.load(path_to_model_checkpoint)
    model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    validation_dir = os.path.join(EVAL_PATH, conf.experiment)
    pred = []
    pred_dict = {}
    for i, data in tqdm(enumerate(test_loader), desc="Evaluation"):
        inputs = data["rgb"]  # currently only gray values
        # print(inputs)

        pred = model(inputs)
        print(pred.tolist())
        # print(pred)

        # Flatten the list
        # flat_list = [item for sublist in pred.tolist() for item in sublist]

        # pred is given back as a pytorch tensor object and cannot be read in with json
        # it must therefore be converted to a list --> pred.tolist()

        # pred = model(xy) - Take model 20240407_030148_model_DummyModel_epoch_0 (should hopefully work). You need to
        # load the model class first (DummyModel in this case) and then add the weights stored in the mentioned file
        # output_dict = get_output_dict(flat_list, data)

        output_dict = get_output_dict(pred.tolist(), data)
        print(output_dict)

        # Append output_dict to pred_dict
        for key, value in output_dict.items():
            if key in pred_dict:
                pred_dict[key].extend(value)
            else:
                pred_dict[key] = value

    # Convert the model output to a dictionary
    # Add the output dictionary to the pred_dict
    # for key in output_dict:
    # pred_dict[key] = output_dict[key]

    print(output_dict)
    print(pred_dict)

    # Write the results to the submission file
    submission_file = os.path.join(validation_dir, "validation_submission.json")
    os.makedirs(os.path.dirname(submission_file), exist_ok=True)
    with open(submission_file, "w") as f:
        json.dump(pred_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, default="Test")
    parser.add_argument("--conf", type=str, default=".\\configs\\eval.yaml")
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
