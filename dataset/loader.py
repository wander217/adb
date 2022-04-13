import math

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
from collections import OrderedDict
import dataset.prep as prep_module
import os
import json
import numpy as np
import cv2 as cv


class DetDataset(Dataset):
    def __init__(self, imgDir: str, imgType: int, tarFile: str, prep: Dict):
        # image dir
        self._imgDir: str = imgDir
        # target file
        self._tarFile: str = tarFile
        # is training
        self._train: bool = 'train' in imgDir
        # image type: 0:jpg, 1:numpy
        self._imgType: int = imgType
        # preprocessing
        self._prep: List = []
        # loading prep module
        if prep is not None:
            for key, item in prep.items():
                cls = getattr(prep_module, key)
                self._prep.append(cls(**item))

        self._imgPath: List = []
        self._target: List = []
        self._loadData()

    def _loadData(self):
        # loading annotation
        with open(self._tarFile, 'r', encoding='utf-8') as file:
            annos = json.loads(file.readline().strip('\n').strip('\r\t').strip())
        # loading image path
        for anno in annos:
            self._imgPath.append(os.path.join(self._imgDir, anno['img']))
            polygons: List = [tar for tar in anno['target']]
            self._target.append(polygons)

    def _loadImage(self, imgPath: str):
        if self._imgType == 1:
            return np.load(imgPath)
        return cv.imread(imgPath)

    def __getitem__(self, index: int, isVisual: bool = False) -> OrderedDict:
        data: OrderedDict = OrderedDict()
        imgPath: str = self._imgPath[index]
        image: np.ndarray = self._loadImage(imgPath)
        data['train'] = self._train
        data['tar'] = self._target[index]
        data['img'] = image
        for proc in self._prep:
            data = proc(data, isVisual)
        if len(self._prep) != 0 and isVisual:
            cv.waitKey(0)
        return data

    def __len__(self):
        return len(self._imgPath)


def padding(img: np.ndarray, shape: Tuple):
    if len(img.shape) == 3:
        c, h, w = img.shape
        new_h, new_w = h, w
        if h > shape[0] or w > shape[1]:
            scale = max([shape[0] / h, shape[1] / w])
            new_h = math.floor(scale * h)
            new_w = math.floor(scale * w)
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        new_image = np.zeros((c, shape[0], shape[1]), dtype=np.uint8)
        new_image[:, :new_h, :new_w] = img
    else:
        h, w = img.shape
        new_image = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        new_h, new_w = h, w
        if h > shape[0] or w > shape[1]:
            scale = max([shape[0] / h, shape[1] / w])
            new_h = math.floor(scale * h)
            new_w = math.floor(scale * w)
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        new_image[:new_h, :new_w] = img
    return new_image


class DetCollate:
    def __init__(self):
        self._limit: int = 1280

    def __call__(self, batch: Tuple) -> OrderedDict:
        imgs: List = []
        probMaps: List = []
        probMasks: List = []
        threshMaps: List = []
        threshMasks: List = []
        polygons: List = []
        ignores: List = []
        output: OrderedDict = OrderedDict()
        mh, mw = 0, 0
        for element in batch:
            c, h, w = element['img'].shape
            mh = max([mh, h])
            mw = max([mw, w])
        mh = min([mh, self._limit])
        mw = min([mw, self._limit])
        for element in batch:
            imgs.append(padding(element['img'], (mh, mw)))
            probMaps.append(padding(element['probMap'], (mh, mw)))
            probMasks.append(padding(element['probMask'], (mh, mw)))
            threshMaps.append(padding(element['threshMap'], (mh, mw)))
            threshMasks.append(padding(element['threshMask'], (mh, mw)))
            if "polygon" in element:
                polygons.append(element['polygon'])
            if "ignore" in element:
                ignores.append(element['ignore'])
        output.update(
            img=torch.from_numpy(np.asarray(imgs, dtype=np.float64)).float(),
            probMap=torch.from_numpy(np.asarray(probMaps, dtype=np.float64)).float(),
            probMask=torch.from_numpy(np.asarray(probMasks, dtype=np.int16)),
            threshMap=torch.from_numpy(np.asarray(threshMaps, dtype=np.float64)).float(),
            threshMask=torch.from_numpy(np.asarray(threshMasks, dtype=np.int16)),
            shape=[mh, mw]
        )
        if len(polygons) != 0:
            output.update(polygon=torch.from_numpy(np.asarray(polygons, dtype=np.int32)))
        if len(ignores) != 0:
            output.update(ignore=torch.from_numpy(np.asarray(ignores, dtype=np.bool)))
        return output


class DetLoader:
    def __init__(self,
                 dataset: Dict,
                 numWorkers: int,
                 batchSize: int,
                 dropLast: bool,
                 shuffle: bool,
                 pinMemory: bool):
        self._dataHolder: DetDataset = DetDataset(**dataset)
        self._numWorkers: int = numWorkers
        self._batchSize: int = batchSize
        self._dropLast: bool = dropLast
        self._shuffle: bool = shuffle
        self._pinMemory: bool = pinMemory
        self._collate = DetCollate()

    def build(self):
        return DataLoader(
            dataset=self._dataHolder,
            batch_size=self._batchSize,
            num_workers=self._numWorkers,
            drop_last=self._dropLast,
            shuffle=self._shuffle,
            pin_memory=self._pinMemory,
            collate_fn=self._collate
        )
