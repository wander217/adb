import torch
from torch import nn, Tensor
from typing import List
from structure.neck.hour_glass import HourGlass


class DBNeck(nn.Module):
    def __init__(self, data_point: tuple, exp: int, layer_num: int, bias: bool = False):
        super().__init__()
        self._ins: nn.ModuleList = nn.ModuleList([
            nn.Conv2d(data_point[i], exp, kernel_size=1, bias=bias)
            for i in range(len(data_point))
        ])
        self._hour_glass: nn.Module = nn.Sequential(*[
            HourGlass(exp, exp) for _ in range(layer_num)
        ])

    def forward(self, feature: List) -> Tensor:
        """
        :param feature: 4 feature with different size: 1/32, 1/16, 1/8, 1/4
        :return: primitive probability map
        """
        # input processing
        for i in range(len(self._ins)):
            feature[i] = self._ins[i](feature[i])
        # up sampling processing
        feature = self._hour_glass(feature)
        return feature[-1]
