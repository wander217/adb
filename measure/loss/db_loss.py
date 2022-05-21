import torch.nn as nn
from typing import Dict, Tuple
from measure.loss.bce_loss import BceLoss
from measure.loss.dice_loss import DiceLoss
from collections import OrderedDict
from torch import Tensor


class DBLoss(nn.Module):
    def __init__(self, probLoss: Dict, binaryLoss: Dict):
        super().__init__()
        self._probLoss: BceLoss = BceLoss(**probLoss)
        self._binaryLoss = DiceLoss(**binaryLoss)

    def __call__(self, pred: OrderedDict, batch: OrderedDict) -> Tuple:
        probDist: Tensor = self._probLoss(pred['probMap'],
                                          batch['probMap'],
                                          batch['probMask'])
        lossDict: OrderedDict = OrderedDict(probLoss=probDist)
        binaryDist: Tensor = self._binaryLoss(pred['binaryMap'],
                                              batch['binaryMap'],
                                              batch['probMask'])
        lossDict.update(binaryLoss=binaryDist)
        loss = probDist + binaryDist / 2
        return loss, lossDict
