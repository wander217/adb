import os.path
import yaml
import torch
from dataset import SynthLoader
from tool import DetLogger, DetAverager, DetCheckpoint
from typing import Dict, Tuple
import torch.optim as optim
import argparse
import warnings
from loss_model import LossModel


class SynthTrainer:
    def __init__(self,
                 lossModel: Dict,
                 train: Dict,
                 optimizer: Dict,
                 checkpoint: Dict,
                 logger: Dict,
                 totalEpoch: int,
                 startEpoch: int,
                 lr: float,
                 **kwargs):
        self._device = torch.device('cpu')
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        self._model: LossModel = LossModel(**lossModel, device=self._device)
        self._train = SynthLoader(**train).build()
        self._checkpoint: DetCheckpoint = DetCheckpoint(**checkpoint)
        self._logger: DetLogger = DetLogger(**logger)
        optimCls = getattr(optim, optimizer['name'])
        self._lr: float = lr
        self._optim: optim.Optimizer = optimCls(**optimizer['args'],
                                                lr=self._lr,
                                                params=self._model.parameters())
        self._totalEpoch: int = totalEpoch + 1
        self._startEpoch: int = startEpoch

    def _load(self):
        stateDict: Tuple = self._checkpoint.load(self._device)
        if stateDict is not None:
            self._model.load_state_dict(stateDict[0])
            self._optim.load_state_dict(stateDict[1])
            self._startEpoch = stateDict[2] + 1

    def train(self):
        self._load()
        self._logger.reportDelimitter()
        self._logger.reportTime("Start")
        self._logger.reportDelimitter()
        self._logger.reportNewLine()
        for i in range(self._startEpoch, self._totalEpoch):
            self._logger.reportDelimitter()
            self._logger.reportTime("Epoch {}".format(i))
            trainRS: Dict = self._trainStep()
            self._save(trainRS, i)
        self._logger.reportDelimitter()
        self._logger.reportTime("Finish")
        self._logger.reportDelimitter()

    def _trainStep(self) -> Dict:
        self._model.train()
        totalLoss: DetAverager = DetAverager()
        probLoss: DetAverager = DetAverager()
        threshLoss: DetAverager = DetAverager()
        binaryLoss: DetAverager = DetAverager()
        for batch in self._train:
            self._optim.zero_grad()
            batchSize: int = batch['img'].size(0)
            pred, loss, metric = self._model(batch)
            loss = loss.mean()
            loss.backward()
            self._optim.step()
            totalLoss.update(loss.item() * batchSize, batchSize)
            threshLoss.update(metric['threshLoss'].item() * batchSize, batchSize)
            binaryLoss.update(metric['binaryLoss'].item() * batchSize, batchSize)
            probLoss.update(metric['probLoss'].item() * batchSize, batchSize)
        return {
            'totalLoss': totalLoss.calc(),
            'threshLoss': threshLoss.calc(),
            'binaryLoss': binaryLoss.calc(),
            'probLoss': probLoss.calc()
        }

    def _save(self, trainRS: Dict, epoch: int):
        self._logger.reportMetric("training", trainRS)
        self._logger.writeFile({
            'training': trainRS
        })
        self._checkpoint.saveCheckpoint(epoch, self._model, self._optim)
        self._checkpoint.saveModel(self._model, epoch)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Training config")
    parser.add_argument("-p", '--path', type=str, help="path of config file")
    parser.add_argument("-a", '--asset', default='', type=str, help="path of asset")
    parser.add_argument("-r", '--resume', default='', type=str, help="path of checkpoint")
    args = parser.parse_args()
    with open(args.path) as f:
        config: Dict = yaml.safe_load(f)
    if args.asset.strip():
        tmp = args.asset.strip()
        config['train']['dataset']['bg_root'] = os.path.join(tmp, "bg/")
        config['train']['dataset']['font_root'] = os.path.join(tmp, "font/")
        config['train']['dataset']['dct_root'] = os.path.join(tmp, "train.txt")
    if args.resume.strip():
        config['checkpoint']['resume'] = args.resume.strip()
    trainer = SynthTrainer(**config)
    trainer.train()
