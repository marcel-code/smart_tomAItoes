import numpy as np
import torch
import torchvision.models as pretrained_models
from omegaconf.errors import ConfigAttributeError, ConfigKeyError

from ..utils.tools import get_loss
from .baseModel import BaseModel


class initialModel(BaseModel):

    def _init(self, conf):
        self.name = "initialModel"

        pretrained_model = pretrained_models.vgg16(pretrained=True)
        pretrained_model.classifier = pretrained_model.classifier[:-1]
        self.backbone = pretrained_model
        self.backbone.requires_grad_(False)

        self.head = ModelHead()

        if conf.model.pretrained_model != "None":
            self.load_model(conf.model.pretrained_model)
        try:
            # TODO Handling of input_shape via OmegaConf
            self.inputShape = conf.model.input_shape
        except ConfigAttributeError:
            print(
                "Key conf.model.input_shape not included in config file! Change either config file or assign another value!"
            )
        except ConfigKeyError:
            print(
                "Key conf.model.input_shape not included in config file! Change either config file or assign another value!"
            )

        self.loss_fn = get_loss("src.models.utils.losses", torch.nn.Module, conf.train.loss)()

        # self.sigmoid_scaling = torch.Tensor()
        self.output = {}

    def _forward(self, x):
        # Model Architecture
        x = self.backbone(x)
        x = self.head(x)
        return x

    def loss(self, pred, ground_truth):
        return self.loss_fn(pred, ground_truth)

    def load_model(self, state_dict):
        self.head.load_state_dict(torch.load(state_dict))


class ModelHead(torch.nn.Module):
    def __init__(self):
        super(ModelHead, self).__init__()

        self.name = "ModelHead"

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(4096, 200)
        self.linear2 = torch.nn.Linear(200, 4)
        self.activation = torch.nn.ReLU()
        self.finalLayer = torch.nn.Linear(4, 4)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # Final Layer output: 4 values (height, fw_plant, number of tomatoes, leaf_area)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

    def loss(self):
        raise NotImplementedError
