import torch
import torchvision.transforms as T

from featup.util import pca, remove_axes
from featup.featurizers.maskclip.clip import tokenize
from pytorch_lightning import seed_everything
import torch
import torch.nn.functional as F
from PIL import Image
from featup.util import norm, unnorm


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
