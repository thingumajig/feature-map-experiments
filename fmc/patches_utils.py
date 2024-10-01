import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def extract_patch(image: Image, x, y, width, height):
  """
  Extracts a patch from the given image.

  Args:
    image: The PIL Image object.
    x: The x-coordinate of the top-left corner of the patch.
    y: The y-coordinate of the top-left corner of the patch.
    width: The width of the patch.
    height: The height of the patch.

  Returns:
    The extracted patch as a PIL Image object.
  """
  patch = image.crop((x, y, x + width, y + height))
  return patch


def extract_patches_numpy(image, patch_size, step):
    """
    Извлекает патчи из изображения заданного размера и с заданным шагом.

    :param image: Исходное изображение (numpy array).
    :param patch_size: Размер патча (высота, ширина).
    :param step: Шаг (по вертикали и горизонтали).
    :return: Список патчей.
    """
    patches = []
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size

    for y in range(0, height - patch_height + 1, step):
        for x in range(0, width - patch_width + 1, step):
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append(patch)

    return patches

def extract_patches_pillow(image: Image, patch_size, step):
    """
    Извлекает патчи из изображения заданного размера и с заданным шагом.

    :param image: Исходное изображение (Pillow Image).
    :param patch_size: Размер патча (ширина, высота).
    :param step: Шаг (по горизонтали и вертикали).
    :return: Список патчей (Pillow Images).
    """
    patches = []
    width, height = image.size
    patch_width, patch_height = patch_size


    for y in range(0, height - patch_height + 1, step):
        for x in range(0, width - patch_width + 1, step):
            patch = image.crop((x, y, x + patch_width, y + patch_height))
            patches.append(patch)

    return patches


def extract_patches_pillow_matrix(image: Image, patch_size, step):
    """
    Извлекает патчи из изображения заданного размера и с заданным шагом по x и y.

    :param image: Исходное изображение (Pillow Image).
    :param patch_size: Размер патча (ширина, высота).
    :param step_x: Шаг по горизонтали.
    :param step_y: Шаг по вертикали.
    :return: Двумерная матрица патчей (numpy array).
    """
    patches = []
    width, height = image.size
    patch_width, patch_height = patch_size

    step_x, step_y = step

    for y in range(0, height - patch_height + 1, step_y):
        row_patches = []
        for x in range(0, width - patch_width + 1, step_x):
            patch = image.crop((x, y, x + patch_width, y + patch_height))
            row_patches.append(np.array(patch))
        patches.append(row_patches)

    return np.array(patches)


def assemble_image(patches, patches_shape):
    # Определяем размеры патча (предполагается, что все патчи одинакового размера)
    _, patch_width, patch_height = patches[0].shape
    n_rows, n_columns = patches_shape
    # Создаем новое изображение с размерами, равными размеру всего изображения
    assembled_image = Image.new(
        'RGB', (n_columns * patch_width, n_rows * patch_height))

    # Вставляем каждый патч в нужное место
    for i in range(n_rows):
        for j in range(n_columns):
            patch = patches[i * n_columns + j]
            assembled_image.paste(transforms.ToPILImage('RGB')(patch), (j * patch_width, i * patch_height))

    return assembled_image



def combine_patches(index_root, wsi_file_name, method, start, magnification, patch_size):
    wsi_stuff_path = index_root / wsi_file_name
    method_index_path = wsi_stuff_path / method

    x, y = start
    patch_width, patch_height = patch_size
    
    x = x*magnification
    y = y*magnification

    new_magnified_image = Image.new('RGB', (patch_width, patch_height), color=(241, 241, 249))

    for pyi in range(0, magnification, 1):
        for pxj in range(0, magnification, 1): 
            patch_file = method_index_path / f'{x+pxj*patch_width}_{y+pyi*patch_height}.png'
            # print(f'[{pxj}, {pyi}]: ({patch_file}) ')
            if patch_file.exists():
                patch = Image.open(patch_file)
                patch = patch.resize((patch_width // magnification+1, patch_height // magnification+1), resample=Image.LANCZOS)
                new_magnified_image.paste(patch, (pxj*patch_width // magnification, pyi*patch_height // magnification))

    return new_magnified_image
