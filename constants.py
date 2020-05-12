import torch


"""
Possible losses and triplet sampling strategies
"""
LOSS_OPTIONS_STR = 'softmax / contrastive / triplet / arcface / center / coco'
TRIPLET_SAMPLING_OPTIONS_STR = 'all / semihard-neg / hardest-neg / hardest-pos-neg'

"""
Default seed for every script
"""
SEED = 124

"""
PyTorch device: GPU if available, CPU otherwise
"""
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
