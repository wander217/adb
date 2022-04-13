import math
import cv2
import numpy as np
from typing import List, OrderedDict


class DetNorm:
    def __init__(self, mean: List, limit: int):
        self.mean: np.ndarray = np.array(mean)
        self.limit: int = limit

    def __call__(self, data: OrderedDict, isVisual: bool = False):
        output: OrderedDict = self._build(data)
        if isVisual:
            self._visual(output)
        return output

    def _visual(self, data: OrderedDict):
        print(data.keys())
        print(data['img'].shape)
        # threshMap = np.uint8(data['threshMap'] * 255)
        # threshMask = np.uint8(data['threshMask'] * 255)
        # cv2.imshow("new thresh mask", threshMask)
        # cv2.imshow("new thresh map", threshMap)
        # probMap = np.uint8(data['probMap'][0] * 255)
        # probMask = np.uint8(data['probMask'] * 255)
        # cv2.imshow("new prob Map", probMap)
        # cv2.imshow("new prob Mask", probMask)

    def _build(self, data: OrderedDict) -> OrderedDict:
        assert 'img' in data
        image: np.ndarray = data['img']
        h, w, c = image.shape
        if h > self.limit or w > self.limit:
            scale = min([self.limit / w, self.limit / h])
            new_h = math.floor(h * scale)
            new_w = math.floor(w * scale)
            data['img'] = cv2.resize(image,
                                     (new_w, new_h),
                                     interpolation=cv2.INTER_CUBIC)
            data['threshMap'] = cv2.resize(data['threshMap'],
                                           (new_w, new_h),
                                           interpolation=cv2.INTER_CUBIC)
            data['probMap'][0] = cv2.resize(data['probMap'][0],
                                         (new_w, new_h),
                                         interpolation=cv2.INTER_CUBIC)
            data['threshMask'] = cv2.resize(data['threshMask'],
                                            (new_w, new_h),
                                            interpolation=cv2.INTER_CUBIC)
            data['probMask'] = cv2.resize(data['probMask'],
                                          (new_w, new_h),
                                          interpolation=cv2.INTER_CUBIC)
        image = (image.astype(np.float64) - self.mean) / 255.
        data['img'] = np.transpose(image, (2, 0, 1))
        return data
