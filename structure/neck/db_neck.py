import torch
from torch import nn, Tensor
from typing import List


class DBNeck(nn.Module):
    def __init__(self, dataPoint: tuple, exp: int, bias: bool = False):
        super().__init__()

        assert len(dataPoint) >= 4, len(dataPoint)
        self.in5: nn.Module = nn.Conv2d(dataPoint[-1], exp, kernel_size=1, bias=bias)
        self.in4: nn.Module = nn.Conv2d(dataPoint[-2], exp, kernel_size=1, bias=bias)
        self.in3: nn.Module = nn.Conv2d(dataPoint[-3], exp, kernel_size=1, bias=bias)
        self.in2: nn.Module = nn.Conv2d(dataPoint[-4], exp, kernel_size=1, bias=bias)

        # Upsampling layer
        self.up5: nn.Module = nn.Upsample(scale_factor=2)
        self.up4: nn.Module = nn.Upsample(scale_factor=2)
        self.up3: nn.Module = nn.Upsample(scale_factor=2)

        expOutput: int = exp // 4
        self.out5: nn.Module = nn.Sequential(
            nn.Conv2d(exp, expOutput, kernel_size=3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='bilinear')
        )
        self.out4: nn.Module = nn.Sequential(
            nn.Conv2d(exp, expOutput, kernel_size=3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        self.out3: nn.Module = nn.Sequential(
            nn.Conv2d(exp, expOutput, kernel_size=3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )
        self.out2: nn.Module = nn.Sequential(
            nn.Conv2d(exp, expOutput, kernel_size=3, padding=1, bias=bias)
        )

    def forward(self, input: List) -> Tensor:
        '''

        :param input: 4 feature with diffirent size: 1/32, 1/16, 1/8, 1/4
        :return: primitive probmap
        '''
        assert len(input) == 4, len(input)

        # input processing
        fin5: Tensor = self.in5(input[3])
        fin4: Tensor = self.in4(input[2])
        fin3: Tensor = self.in3(input[1])
        fin2: Tensor = self.in2(input[0])

        # upsampling processing
        fup4: Tensor = self.up5(fin5) + fin4
        fup3: Tensor = self.up4(fup4) + fin3
        fup2: Tensor = self.up3(fup3) + fin2

        # Output processing
        fout5: Tensor = self.out5(fin5)
        fout4: Tensor = self.out4(fup4)
        fout3: Tensor = self.out3(fup3)
        fout2: Tensor = self.out2(fup2)

        # Concatenate
        fusion: Tensor = torch.cat([fout5, fout4, fout3, fout2], 1)
        return fusion
