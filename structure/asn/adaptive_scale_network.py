import math
import time
import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from typing import List
import torch.nn.functional as F


def resize(x: Tensor, org_size: List):
    return F.interpolate(x, org_size, mode='bilinear', align_corners=True)


class ConvNormActivation(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 activation: nn.Module = None,
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1,
                 bias: bool = False):
        super().__init__()
        self._conv: nn.Module = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias),
            nn.BatchNorm2d(output_channel))
        self._act: nn.Module = activation

    def forward(self, x: Tensor):
        y: Tensor = self._conv(x)
        if self._act is not None:
            y = self._act(y)
        return y


class ResidualConv(nn.Module):
    def __init__(self,
                 input_channel: int,
                 hidden_channel: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1):
        super().__init__()
        self._conv: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,
                      out_channels=hidden_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channel,
                      out_channels=input_channel,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(input_channel))

    def forward(self, x: Tensor):
        y: Tensor = self._conv(x) + x
        return y


def weight_int(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1.)
        module.bias.data.fill_(1e-4)


class AdaptiveScaleNetwork(nn.Module):
    def __init__(self, shape: List):
        super().__init__()
        self._shape: List = shape
        self._lower: Tensor = (self._shape[0] - 1) * (self._shape[1] - 1)
        self._weight_init: nn.Module = nn.Sequential(
            ConvNormActivation(input_channel=3,
                               output_channel=16,
                               activation=nn.LeakyReLU(inplace=True)),
            ConvNormActivation(input_channel=16,
                               output_channel=1,
                               activation=nn.Hardsigmoid(inplace=True)))
        self._residual: nn.Module = nn.Sequential(
            ConvNormActivation(input_channel=3,
                               output_channel=16,
                               activation=nn.LeakyReLU(inplace=True)),
            ResidualConv(input_channel=16, hidden_channel=16),
            ResidualConv(input_channel=16, hidden_channel=16),
            ResidualConv(input_channel=16, hidden_channel=16),
            ConvNormActivation(input_channel=16, output_channel=16))
        self._conv: nn.Module = ConvNormActivation(input_channel=16, output_channel=3)

    def forward(self, x: Tensor):
        y = resize(self._weight_init(x) * x, self._shape)
        y = self._conv(self._residual(y)) #+ resize(x, self._shape)
        return y


if __name__ == "__main__":
    conv = AdaptiveScaleNetwork([640, 640])
    img = cv2.imread(r'D:\workspace\project\db_pp\test_image\test2_1.png')
    h, w, c = img.shape
    new_image = np.zeros((math.ceil(h / 32) * 32, math.ceil(w / 32) * 32, 3), dtype=np.uint8)
    new_image[:h, :w, :] = img
    new_h, new_w, _ = new_image.shape
    input = torch.from_numpy(new_image).permute(2, 0, 1).unsqueeze(0)
    start = time.time()
    new_image = conv(input.float()).squeeze().permute(1, 2, 0).cpu().detach().numpy()
    # new_image = cv2.resize(np.uint8(new_image), (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(np.uint8(img), (640, 640), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('aaa', np.uint8(new_image))
    cv2.imshow('bbb', np.uint8(img))
    cv2.waitKey(0)
    total_params = sum(p.numel() for p in conv.parameters())
    print(total_params)
    print(time.time() - start)
