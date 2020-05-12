import os
from os.path import isdir, join
import torch
import numpy as np
import random
import constants


def set_custom_seed(seed: int = None):
    """
    Set a fixed seed for the experiment.

    :return: nothing
    """
    if seed is None:
        seed = constants.SEED
    print(f"[Seed: {seed}]")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_log_dir(exp_path: str, args) -> str:
    """Create the directory where logs, models, plots and
    other experiment related files will be stored.

    :param exp_path: the path to the log directory
    :param args: an object containing:
        - model: name of the model
        - loss: name of the loss
        - lr: learning rate value
        - margin: margin value
        - loss_scale: loss scale or weight value (AAM, CoCo, Center)
        - seed: seed value
    :return: the name of the created directory, or exit the program if the directory exists
    """
    log_path = f"{exp_path}/{args.model}/{args.loss}/lr={args.lr}"
    if args.loss in ['arcface', 'contrastive']:
        log_path = join(log_path, f"margin={args.margin}")
    if args.loss == 'center':
        log_path = join(log_path, f"lambda={args.loss_scale}")
    if args.loss in ['arcface', 'coco', 'triplet']:
        log_path = join(log_path, f"scale={args.loss_scale}")
    log_path = join(log_path, f"seed={args.seed}")
    if isdir(log_path):
        print(f"The experience directory '{log_path}' already exists")
        exit(1)
    os.makedirs(log_path)
    return log_path


def dump_params(filepath: str, args):
    """Dump all script arguments to a file.

    :param filepath: path to the output file
    :param args: arguments received by the script
    :return: nothing
    """
    with open(filepath, 'w') as out:
        for k, v in sorted(vars(args).items()):
            out.write(f"{k}={v}\n")


def model_size(model) -> float:
    """Calculate the number of parameters in a model (in millions).

    :param model: `nn.Module`
    :return: `float`, number of parameters in millions
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000