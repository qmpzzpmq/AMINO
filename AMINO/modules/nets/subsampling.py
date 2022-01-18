#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""Subsampling layer definition."""

from typing import Tuple

import torch


class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-12),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
            self,
            xs: torch.Tensor,
            xs_len: torch.Tensor,
            x_mask: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .
        """
        xs = self.out(xs)
        xs, pos_emb = self.pos_enc(xs, offset)
        return xs, xs_len, pos_emb, x_mask,