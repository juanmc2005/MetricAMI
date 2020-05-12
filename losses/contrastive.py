#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from distances import Distance


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss module
    Reference: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    :param device: a device where to execute the calculations
    :param margin: the margin to separate feature vectors considered different
    :param distance: a Distance object to calculate our Dw
    """
    
    def __init__(self, device: str, margin: float, distance: Distance, size_average: bool, online: bool):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.margin = margin
        self.distance = distance
        self.size_average = size_average
        self.online = online
    
    def forward(self, feat, logits, y):
        """
        Calculate the contrastive loss
        :param feat: a tensor corresponding to a batch of size (N, d), where
            N = batch size
            d = dimension of the feature vectors
        :param logits: unused, it's been kept for compatibility purposes
        :param y: a non one-hot label tensor corresponding to the batch
        :return: the contrastive loss
        """
        if self.online:
            # First calculate the distances between every sample in the batch
            nbatch = feat.size(0)
            dist = self.distance.pdist(feat).to(self.device)
            # Calculate the ground truth Y corresponding to the pairs
            gt = []
            for i in range(nbatch-1):
                for j in range(i+1, nbatch):
                    gt.append(int(y[i] != y[j]))
            gt = torch.Tensor(gt).float().to(self.device)
        else:
            feat1, feat2 = feat
            dist = self.distance.dist(feat1, feat2)
            gt = y
        # Calculate the losses as described in the paper
        # Removing squares as cosine distance is in [0,2]
        loss = (1 - gt) * dist + gt * torch.clamp(self.margin - dist, min=1e-8)
        loss = torch.sum(loss) / 2
        # Average by batch size if requested
        return loss / dist.size(0) if self.size_average else loss
