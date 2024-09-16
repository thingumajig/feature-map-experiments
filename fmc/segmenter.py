from pathlib import Path
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

def save_indexes(source_image_key, hr_feats_pca, patches_mat_shape, workspace_dir, method_name='pca', ):
    indexes_path = workspace_dir / '.indexes'
    indexes_path.mkdir(parents=True, exist_ok=True)

    current_indexes_path = indexes_path / source_image_key
    current_indexes_path.mkdir(exist_ok=True)


    _, patch_width, patch_height = hr_feats_pca[0].shape
    n_rows, n_columns = patches_mat_shape
    rows = 0 
    cols = 0
    for i in range(n_rows):
        for j in range(n_columns):
            patch = hr_feats_pca[i * n_columns + j]
            pi = T.ToPILImage('RGB')(patch)
            pi.save(current_indexes_path / f'{method_name}_{j*patch_height}_{i*patch_width}.png')


def umap_fl(image_feats_list, dim=3, fit_umap=None, max_samples=None, n_jobs=70):
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
        print('Create new umap')
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


from sklearn.cluster import KMeans

def kmeans_color_quantization(pixel_values, k=3):
    # Параметры для KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Применение KMeans
    kmeans.fit(pixel_values)

    # Преобразование центров обратно в uint8
    centers = kmeans.cluster_centers_

    return kmeans, centers


def kmeans_predict(kmeans, hr_feats, palette = None):
    shape = hr_feats.shape
    feats = hr_feats.reshape((-1, shape[-1]))
    # print(f'{shape=} {feats.shape=}')

    labels = kmeans.predict(feats)
    # print(f'{labels.shape=}')
    centers = kmeans.cluster_centers_
    if palette is not None:
        segmented_image = palette[labels.flatten()]
    else:
        segmented_image = centers[labels.flatten()]
    # print(f'{segmented_image.shape=}')
    segmented_image = segmented_image.reshape((shape[0],shape[1],3))
    return segmented_image

def kmeans_process(hr_feats_flat, k = 15):
    all_feats_values = hr_feats_flat.reshape(-1, 3)
    
    kmeans, centers = kmeans_color_quantization(all_feats_values, k)

    all_segmented_feats = []
    for hr_feats in hr_feats_flat:
        segmented_feats = kmeans_predict(kmeans, hr_feats)
        all_segmented_feats.append(segmented_feats)
        
    return all_segmented_feats