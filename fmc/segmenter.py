import torch
import torchvision.transforms as T

from featup.util import pca, remove_axes
from featup.featurizers.maskclip.clip import tokenize
from pytorch_lightning import seed_everything
import torch
import torch.nn.functional as F
from PIL import Image
from featup.util import norm, unnorm

import numpy as np
import umap


use_norm = True
transform = T.Compose([
    T.Resize(224),
    T.CenterCrop((224, 224)),
    T.ToTensor(),
    norm
])


def init_upsampler(device='cuda'):
    return torch.hub.load("mhamilton723/FeatUp",
                            'dinov2', use_norm=use_norm).to(device)


def mass_upsample(upsampler, flat_patches_mat, batch_size=4, device='cuda'):
    image_tensors = [transform(Image.fromarray(p)) for p in flat_patches_mat]
    image_tensors = torch.stack(image_tensors).to(device)
    print(f'{image_tensors.shape=}')

    hr_feats_list = []
    lr_feats_list = []
    num_elements = image_tensors.shape[0]

    for start in range(0, num_elements, batch_size):
        end = min(start + batch_size, num_elements)
        batch = image_tensors[start:end]
        # print(f'{batch.shape=}')

        with torch.no_grad():
            hr_feats_list.append(upsampler(batch).cpu())
            lr_feats_list.append(upsampler.model(batch).cpu())

        torch.cuda.empty_cache()

    lr_feats_list = torch.cat(lr_feats_list, dim=0)
    hr_feats_list = torch.cat(hr_feats_list, dim=0)

    print(f'{lr_feats_list.shape=}  {hr_feats_list.shape=}')

    return lr_feats_list, hr_feats_list



def umap_fl(image_feats_list, dim=3, fit_umap=None, max_samples=None, n_jobs=60):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_umap is None:
            tensor = F.interpolate(
                tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_umap is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    print(f'umap fit: {x.shape=}')
    if fit_umap is None:
        fit_umap = umap.UMAP(n_components=dim, 
                            #  random_state=42, 
                             n_jobs=n_jobs, verbose=True,
                             ).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_umap.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(
            x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device)
            )

    return reduced_feats, fit_umap