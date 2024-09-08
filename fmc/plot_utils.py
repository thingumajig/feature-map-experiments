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


def display_image_grid(flat_images, grid_size, figsize=(10, 10)):
    """
    Выводит матрицу изображений заданного размера.

    :param images: Список изображений (numpy arrays).
    :param grid_size: Размер сетки (строки, столбцы).
    :param figsize: Размер фигуры.
    """
    num_rows, num_cols = grid_size

    # grid = plt.GridSpec(num_rows, num_cols, wspace=0.0, hspace=0.0)


    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # plt.subplots_adjust(wspace=0.2, hspace=0)

    for i, ax in enumerate(axes.flat):
        if i < len(flat_images):
            ax.axis('off')
            ax.imshow(flat_images[i], aspect=1.)
        else:
            ax.axis('off')

    plt.tight_layout(pad=0.01, h_pad=0.1, w_pad=0.2)

    plt.show()


def display_image_grid_gs(flat_images, grid_size, figsize=(12, 12)):
    """
    Выводит матрицу изображений заданного размера.

    :param images: Список изображений (numpy arrays).
    :param grid_size: Размер сетки (строки, столбцы).
    :param figsize: Размер фигуры.
    """
    num_rows, num_cols = grid_size

    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(num_rows, num_cols, figure=fig, # width_ratios=[1, 1, 1],
         wspace=0.05, hspace=0.01, top=0.95, bottom=0.05, left=0.17, right=0.845)

    for i in range(num_rows):
        for j in range(num_cols):
            ax = plt.subplot(grid[i, j])
            if i * num_cols + j < len(flat_images):
                ax.axis('off')
                ax.imshow(flat_images[i * num_cols + j], aspect='auto')
            else:
                ax.axis('off')

    # plt.tight_layout(pad=0.01, h_pad=0.1, w_pad=0.2)

    plt.show()
