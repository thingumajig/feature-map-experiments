from featup.util import pca, remove_axes
from featup.featurizers.maskclip.clip import tokenize
from pytorch_lightning import seed_everything
import torch
import torch.nn.functional as F


def init_upsampler(use_norm = True, device='cuda'):
    return torch.hub.load("mhamilton723/FeatUp",
                            'dinov2', use_norm=use_norm).to(device)