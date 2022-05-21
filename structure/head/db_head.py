import torch
from torch import nn, Tensor
from collections import OrderedDict
from typing import List
import torch.nn.functional as F


class DBHead(nn.Module):
    def __init__(self, exp: int, thresh: int):
        super().__init__()
        self.thresh: int = thresh

        exp_output: int = exp // 4
        self.prob: nn.Module = nn.Sequential(
            nn.Conv2d(exp, exp_output, kernel_size=3, padding=1),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, exp_output, kernel_size=2, stride=2),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        self.thresh: nn.Module = nn.Sequential(
            nn.Conv2d(exp, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def resize(self, x: Tensor, shape: List):
        return F.interpolate(x, shape, mode="bilinear", align_corners=True)

    def binarization(self, probMap: Tensor, thresh: Tensor):
        return torch.reciprocal(1. + torch.exp(-50 * (probMap - thresh)))

    def forward(self, x: Tensor, shape: List) -> OrderedDict:
        result: OrderedDict = OrderedDict()
        # calculate probability map
        probMap: Tensor = self.resize(self.prob(x), shape)
        thresh: Tensor = self.thresh(x)
        binaryMap: Tensor = self.binarization(probMap, thresh)
        binaryMap = F.max_pool2d(binaryMap.float(), 9, 1, 4)
        result.update(probMap=probMap, binaryMap=binaryMap)
        return result
