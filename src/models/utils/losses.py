import torch


class DummyLoss(torch.nn.Module):
    def __init__(self):
        super(DummyLoss, self).__init__()

    def forward(self, inputs, targets):
        loss = -1 * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        return loss.mean()


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets):
        # TODO Implementation of custom loss function
        raise NotImplementedError
