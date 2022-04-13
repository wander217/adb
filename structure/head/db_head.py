from torch import nn, Tensor
from collections import OrderedDict
import torch


class DBHead(nn.Module):
    def __init__(self, k: int, exp: int, bias: bool = False):
        super().__init__()
        self.k: int = k

        exp_output: int = exp // 4
        self.prob: nn.Module = nn.Sequential(
            nn.Conv2d(exp, exp_output, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, exp_output, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, 1, kernel_size=2, stride=2, bias=bias),
            nn.Sigmoid()
        )
        self.thresh: nn.Module = nn.Sequential(
            nn.Conv2d(exp, exp_output, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, exp_output, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(exp_output),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(exp_output, 1, kernel_size=2, stride=2, bias=bias),
            nn.Sigmoid()
        )

    def binarize(self, probMap: Tensor, threshMap: Tensor):
        return torch.reciprocal(1. + torch.exp(-self.k * (probMap - threshMap)))

    def forward(self, x: Tensor) -> OrderedDict:
        result: OrderedDict = OrderedDict()
        # calculate probability map
        probMap: Tensor = self.prob(x)
        result.update(probMap=probMap)
        threshMap: Tensor = self.thresh(x)
        binaryMap: Tensor = self.binarize(probMap, threshMap)
        result.update(binaryMap=binaryMap, threshMap=threshMap)
        return result
