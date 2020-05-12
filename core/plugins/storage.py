import torch
import torch.nn as nn
from os.path import join
import core.base as base
from core.optim import Optimizer
from models import MetricNet
import constants


class ModelSaver:

    def __init__(self, loss_name: str):
        self.loss_name = loss_name

    def save(self, epoch: int, model: MetricNet, loss_fn: nn.Module,
             optim: Optimizer, accuracy: float, filepath: str):
        print(f"Saving model to {filepath}")
        torch.save({
            'epoch': epoch,
            'trained_loss': self.loss_name,
            'common_state_dict': model.encoder_state_dict(),
            'loss_module_state_dict': model.classifier.state_dict() if model.classifier is not None else None,
            'loss_state_dict': loss_fn.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'accuracy': accuracy
        }, filepath)


class ModelLoader:

    def __init__(self, path: str, restore_optimizer: bool = True):
        self.path = path
        self.restore_optimizer = restore_optimizer

    def get_trained_loss(self):
        return torch.load(self.path, map_location=constants.DEVICE)['trained_loss']

    def restore(self, model: MetricNet, loss_fn: nn.Module, optimizer: Optimizer, current_loss: str):
        checkpoint = torch.load(self.path, map_location=constants.DEVICE)
        model.load_encoder_state_dict(checkpoint['common_state_dict'])
        if current_loss == checkpoint['trained_loss']:
            loss_fn.load_state_dict(checkpoint['loss_state_dict'])
            if model.classifier is not None:
                model.classifier.load_state_dict(checkpoint['loss_module_state_dict'])
            if self.restore_optimizer:
                optimizer.load_state_dict(checkpoint['optim_state_dict'])
        print(f"Recovered Model. Epoch {checkpoint['epoch']}. Dev Metric {checkpoint['accuracy']}")
        return checkpoint

    def load(self, model: MetricNet, current_loss: str):
        checkpoint = torch.load(self.path, map_location=constants.DEVICE)
        model.load_encoder_state_dict(checkpoint['common_state_dict'])
        if current_loss == checkpoint['trained_loss'] and model.classifier is not None:
            model.classifier.load_state_dict(checkpoint['loss_module_state_dict'])
        return checkpoint


class BestModelSaver(base.TestListener):

    def __init__(self, base_path: str, loss_name: str):
        super(BestModelSaver, self).__init__()
        self.base_path = base_path
        self.saver = ModelSaver(loss_name)

    def on_best_accuracy(self, epoch, model, loss_fn, optim, accuracy, feat, y):
        self.saver.save(epoch, model, loss_fn, optim, accuracy,
                        join(self.base_path, "best.pt"))
