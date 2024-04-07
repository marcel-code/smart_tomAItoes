import torch
from omegaconf.errors import ConfigAttributeError, ConfigKeyError

from ..utils.tools import get_loss
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

        self.model = self._generate_model_layers()

        self.loss_fn = get_loss("src.models.utils.losses", torch.nn.Module, conf.train.loss)

    def _generate_model_layers(self):
        self.flatten = torch.nn.Flatten()
        if len(self.inputShape) > 2:
            self.linear1 = torch.nn.Linear(self.inputShape[0] * self.inputShape[1] * self.inputShape[2], 200)
        elif len(self.inputShape) == 2:
            self.linear1 = torch.nn.Linear(self.inputShape[0] * self.inputShape[1], 200)
        else:
            raise NotImplementedError("Case not implemented in DummyModel")

        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 2)
        self.softmax = torch.nn.Softmax()

    def _forward(self, x):
        # Model Architecture
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

    def loss(self, pred, data):
        return self.loss_fn(pred, data)
