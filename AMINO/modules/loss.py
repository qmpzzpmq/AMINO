"""Label smoothing module."""

from typing import OrderedDict

import torch
from torch import nn

from AMINO.utils.init_object import init_object

IGNORE_ID = -1

class LABEL_SMOOTHING_LOSS(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        smoothing (float): smoothing rate (0.0 means the conventional CE)
    """
    def __init__(
        self,
        size: int,
        smoothing: float,
        reduction: str = "none",
    ):
        """Construct an LabelSmoothingLoss object."""
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        if reduction == "none":
            self.reduction = torch.nn.Identity()
        elif reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.
        Args:
            x (torch.Tensor): prediction (batch, class)
            target (torch.Tensor): target onehot (batch, class)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        assert x.size(-1) == self.size
        assert target.size(-1) == self.size
        target = target.to(dtype=torch.float)
        mask = (target == 0.0)
        target = target.masked_fill(mask, self.smoothing)
        target = target.masked_fill(~mask, self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), target)
        return self.reduction(kl)


class AMINO_LOSSES(nn.Module):
    def __init__(
        self,
        nets,
        weights,
    ):
        super().__init__()
        assert list(nets.keys()) == list(weights.keys()), \
            f"the key of loss weight is not same with the key in loss net"
        self.nets = OrderedDict()
        for k, v in nets.items():
            self.nets[k] = init_object(v)
        self.nets = nn.ModuleDict(self.nets)
        self.weights = dict()
        for k, v in weights.items():
            self.weights[k] = torch.tensor(v, requires_grad=False)

    def forward(self, pred_dict, target_dict, len_dict):
        loss_dict = dict()
        loss_dict["total"] = torch.tensor(
            0.0, device=list(self.weights.values())[0].device,
        )
        for key in pred_dict.keys():
            loss_dict[key] = self.nets[key](
                pred_dict[key], target_dict[key]
            ).sum() / len_dict[key]
            loss_dict["total"] += loss_dict[key] * self.weights[key]
        return loss_dict
