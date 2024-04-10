import torch


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


def get_output_dict(pred, data, key_list=["height", "fw_plant", "leaf_area", "number_of_red_fruits"]):
    """Conversion of model output () to dict"""
    # TODO Check for correctness
    res = {}
    for i in range(len(data["name"])):
        res[data["name"][i]] = {}
        for j in range(len(key_list)):
            res[data["name"][i]][key_list[j]] = pred[i][j]
    return res
