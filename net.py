#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 9:57
# @Author  : Ws
# @File    : net.py
# @Software: PyCharm
import os

import lightning as L
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch import nn
from matplotlib import pyplot as plt

from RWKV3D.model.unet import SwinUNETR


class Net(L.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.net = SwinUNETR(
            in_channels=1,
            out_channels=1,
            patch_size=(2, 2, 1)
        )
        self.compute_loss = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch['model'])
        loss = self.compute_loss(out, batch['data'])

        self.log_dict({'train_loss': loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch['model'])
        loss = self.compute_loss(out, batch['data'])

        self.log_dict({'val_loss': loss}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9,
                                                                       0.95))
