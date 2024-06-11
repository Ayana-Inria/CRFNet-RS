# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:34:32 2022

@author: marti
"""

import torch.nn as nn 
from utils.utils_network import *

class CRFNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(CRFNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.fc1 = OutConv(256, n_classes)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.fc2 = OutConv(128, n_classes)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.fc3 = OutConv(64, n_classes)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.neigh = NeighConv(n_classes, n_classes)

    def forward(self, x):

        activations = []
        out_fc = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        out1 = self.fc1(x)
        out_fc.append(out1)
        activations.append(x)
        x = self.up2(x, x3)
        out2 = self.fc2(x)
        out_fc.append(out2)
        activations.append(x)
        x = self.up3(x, x2)
        out3 = self.fc3(x)
        out_fc.append(out3)
        activations.append(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        neigh = self.neigh(logits)[:,:,:-2, :-2]
        return logits, out_fc, neigh, activations