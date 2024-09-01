# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:34:32 2022

@author: marti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
import numpy as np



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class NeighConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.neigh_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2) # this can be substituted by a depthwise convolution
    
    def forward(self, x):
        with torch.no_grad():
            # self.neigh_conv.weight[:, :, 0, 0] = 0    # uncomment for + neighborhood 
            # self.neigh_conv.weight[:, :, 2, 0] = 0
            # self.neigh_conv.weight[:, :, 0, 2] = 0
            # self.neigh_conv.weight[:, :, 2, 2] = 0
        return self.neigh_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

def compute_class_weight(labels):
    
    class_weights = np.zeros(6,) # 6 = number of labels, good for Vaihingen and Potsdam, 8 for Zeebruges
    class_freq = {}
    for label in np.unique(labels):
        class_freq[label] = np.sum(labels==label)
    
    max_value = max(class_freq.items(), key=operator.itemgetter(1))[1]   #obtain the largest class frequency
    
    for label in np.unique(labels):
        if label != 6:
            class_weights[label] = max_value / class_freq[label]
    
    return torch.from_numpy(class_weights).float()
