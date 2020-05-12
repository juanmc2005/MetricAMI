#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import numpy as np
import torch


class SimDatasetPartition:

    def __iter__(self):
        return self

    def nbatches(self) -> int:
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class SimDataset:

    def training_partition(self) -> SimDatasetPartition:
        raise NotImplementedError

    def dev_partition(self) -> SimDatasetPartition:
        raise NotImplementedError

    def test_partition(self) -> SimDatasetPartition:
        raise NotImplementedError


class TextPartition(SimDatasetPartition):

    def __init__(self, data, batch_size: int, train: bool, batches_per_epoch: int = None):
        self.data = data
        self.batch_size = batch_size
        self.train = train
        self.batches_per_epoch = batches_per_epoch
        self.generator = self._generate()

    def _generate(self):
        while True:
            if self.train:
                np.random.shuffle(self.data)
            for i in range(0, len(self.data), self.batch_size):
                j = min(i + self.batch_size, len(self.data))
                yield self.data[i:j]

    def nbatches(self):
        total_batches = int(math.ceil(len(self.data) / self.batch_size))
        if self.batches_per_epoch is not None and self.batches_per_epoch in range(1, total_batches):
            return self.batches_per_epoch
        else:
            return total_batches

    def __next__(self):
        batch = next(self.generator)
        x, y = [x for x, _ in batch], torch.Tensor([y for _, y in batch]).long()
        return x, y
