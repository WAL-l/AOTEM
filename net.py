#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 9:57
# @Author  : Ws
# @File    : net.py
# @Software: PyCharm
import os

import lightning as L
import torch
from torch import nn

from model.model import UNET


class Net(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.net = UNET(
            in_channels=1,
            out_channels=1,
            input_shape=(64, 64, 32),
            hidden_size=384,
            depth=12,
            num_heads=6,
            patch_size=2,
            mask_threshold=0
        )
        self.compute_loss = nn.MSELoss()
        self.lr = lr

    def forward(self, x, height):
        return self.net(x, height)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch['model'], batch['height'])
        loss = self.compute_loss(out, batch['data'])

        self.log_dict({'train_loss': loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch['model'], batch['height'])
        loss = self.compute_loss(out, batch['data'])
        self.log_dict(
            {'val_loss': loss}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9,
                                                                       0.95))
