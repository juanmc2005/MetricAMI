#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MetricNet(nn.Module):

    def __init__(self, encoder: nn.Module, classifier: nn.Module = None):
        super(MetricNet, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def encoder_state_dict(self):
        return self.encoder.state_dict()

    def load_encoder_state_dict(self, checkpoint):
        self.encoder.load_state_dict(checkpoint)

    def forward(self, x, y):
        x = self.encoder(x)
        logits = self.classifier(x, y) if self.classifier is not None else None
        return x, logits

    def all_params(self):
        params = [self.encoder.parameters()]
        if self.classifier is not None:
            params.append(self.classifier.parameters())
        return params
