import torch
from omegaconf.errors import ConfigAttributeError, ConfigKeyError

from src.models.utils.losses import DummyLoss

from .baseModel import BaseModel


class DummyModel(BaseModel):

    def _init(self, conf):
        self.name = "DummyModel"
        try:
            # TODO Handling of input_shape via OmegaConf
            inputShape = conf.model.input_shape
        except ConfigAttributeError:
            print(
                "Key conf.model.input_shape not included in config file! Change either config file or assign another value!"
            )
        except ConfigKeyError:
            print(
                "Key conf.model.input_shape not included in config file! Change either config file or assign another value!"
            )

        self.inputShape = inputShape
        self.flatten = torch.nn.Flatten()
        if len(inputShape) > 2:
            self.linear1 = torch.nn.Linear(inputShape[0] * inputShape[1] * inputShape[2], 200)
        elif len(inputShape) == 2:
            self.linear1 = torch.nn.Linear(inputShape[0] * inputShape[1], 200)
        else:
            raise NotImplementedError("Case not implemented in DummyModel")

        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 2)
        self.softmax = torch.nn.Softmax()
        self.loss_fn = DummyLoss()

    def _forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x

    def loss(self, pred, data):
        return self.loss_fn(pred, data)
