import matplotlib.pyplot as plt
from featup.util import pca, remove_axes
from featup.featurizers.maskclip.clip import tokenize
from pytorch_lightning import seed_everything
import torch
import torch.nn.functional as F


@torch.no_grad()
def plot_feats_pca(image, lr, hr, fit_pca = None):
    assert len(image.shape) == len(lr.shape) == len(hr.shape) == 3
    seed_everything(0)
    [lr_feats_pca, hr_feats_pca], fit_pca = pca([lr.unsqueeze(0), hr.unsqueeze(0)], fit_pca = fit_pca)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[0].set_title("Image")
    ax[1].imshow(lr_feats_pca[0].permute(1, 2, 0).detach().cpu())
    ax[1].set_title("Original Features")
    ax[2].imshow(hr_feats_pca[0].permute(1, 2, 0).detach().cpu())
    ax[2].set_title("Upsampled Features")
    remove_axes(ax)
    plt.show()
    return fit_pca


def display_image_grid(flat_images, grid_size, figsize=(12, 12)):
    """
    Выводит матрицу изображений заданного размера.

    :param images: Список изображений (numpy arrays).
    :param grid_size: Размер сетки (строки, столбцы).
    :param figsize: Размер фигуры.
    """
    num_rows, num_cols = grid_size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    for i, ax in enumerate(axes.flat):
        if i < len(flat_images):
            ax.imshow(flat_images[i])
            ax.axis('off')
        else:
            ax.axis('off')

    # plt.tight_layout(h_pad=0.001, w_pad=0.2)
    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.show()
