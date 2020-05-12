# -*- coding: utf-8 -*-
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as sch
import constants
from distances import CosineDistance
from losses.center import CenterLinear, SoftmaxCenterLoss
from losses.wrappers import LossWrapper
from losses.arcface import ArcLinear
from losses.coco import CocoLinear
from losses.contrastive import ContrastiveLoss
from losses.triplet import TripletLoss, BatchAll
import core.base as base
from models import MetricNet


def reduce_lr(opts):
    def get_sched(o):
        return sch.ReduceLROnPlateau(o, mode='max', factor=0.5,
                                     patience=5, verbose=True)
    return [get_sched(opt) for opt in opts]


def split_optimizers(model: MetricNet, lr: float):
    params = model.all_params()
    return [optim.RMSprop(params[0], lr=lr),
            optim.RMSprop(params[1], lr=10 * lr)]


class LossConfig:

    def __init__(self, test_distance):
        self.test_distance = test_distance

    @property
    def loss(self):
        raise NotImplementedError

    @property
    def clf(self):
        raise NotImplementedError

    def get_optimizers(self, model: MetricNet, lr: float):
        raise NotImplementedError

    def optimizer(self, model: MetricNet, lr: float):
        optimizers = self.get_optimizers(model, lr)
        return base.Optimizer(optimizers, reduce_lr(optimizers))


class SoftmaxConfig(LossConfig):

    def __init__(self, nfeat, nclass):
        super().__init__(CosineDistance())
        self.nfeat, self.nclass = nfeat, nclass

    @property
    def loss(self):
        return LossWrapper(nn.NLLLoss().to(constants.DEVICE))

    @property
    def clf(self):
        return CenterLinear(self.nfeat, self.nclass)

    def get_optimizers(self, model: MetricNet, lr: float):
        return [optim.RMSprop(model.parameters(), lr=lr)]


class ArcFaceConfig(LossConfig):

    def __init__(self, nfeat, nclass, margin, s):
        super().__init__(CosineDistance())
        self.clf_ = ArcLinear(nfeat, nclass, margin, s)

    @property
    def loss(self):
        return LossWrapper(nn.CrossEntropyLoss().to(constants.DEVICE))

    @property
    def clf(self):
        return self.clf_

    def get_optimizers(self, model: MetricNet, lr: float):
        return split_optimizers(model, lr)


class CenterConfig(LossConfig):

    def __init__(self, nfeat, nclass, lweight=1, distance=CosineDistance()):
        super().__init__(distance)
        self.nfeat, self.nclass = nfeat, nclass
        self.loss_ = SoftmaxCenterLoss(constants.DEVICE, nfeat, nclass, lweight, distance)

    @property
    def loss(self):
        return self.loss_

    @property
    def clf(self):
        return CenterLinear(self.nfeat, self.nclass)

    def get_optimizers(self, model: MetricNet, lr: float):
        params = model.all_params()
        return [optim.RMSprop(params[0], lr=lr),
                optim.RMSprop(params[1], lr=10 * lr)]


class CocoConfig(LossConfig):

    def __init__(self, nfeat, nclass, alpha):
        super().__init__(CosineDistance())
        self.clf_ = CocoLinear(nfeat, nclass, alpha)

    @property
    def loss(self):
        return LossWrapper(nn.CrossEntropyLoss().to(constants.DEVICE))

    @property
    def clf(self):
        return self.clf_

    def get_optimizers(self, model: MetricNet, lr: float):
        return split_optimizers(model, lr)


class ContrastiveConfig(LossConfig):

    def __init__(self, margin=0.2, distance=CosineDistance(), size_average=True, online=True):
        super().__init__(distance)
        self.loss_ = ContrastiveLoss(constants.DEVICE, margin, distance, size_average, online)

    @property
    def loss(self):
        return self.loss_

    @property
    def clf(self):
        return None

    def get_optimizers(self, model: MetricNet, lr: float):
        return [optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)]


class TripletConfig(LossConfig):

    def __init__(self, scaling: float, distance=CosineDistance(),
                 size_average: bool = True, online: bool = True, sampling=BatchAll()):
        super().__init__(distance)
        self.loss_ = TripletLoss(constants.DEVICE, scaling, distance,
                                 size_average, online, sampling)

    @property
    def loss(self):
        return self.loss_

    @property
    def clf(self):
        return None

    def get_optimizers(self, model: MetricNet, lr: float):
        return [optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)]
