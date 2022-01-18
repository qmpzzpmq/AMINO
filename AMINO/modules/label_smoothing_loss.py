#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Label smoothing module."""

import torch
from torch import nn

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


# class AMINO_LOSS(nn.Module):
#     def __init__()