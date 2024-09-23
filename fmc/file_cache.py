import io
from unittest.mock import patch
import numpy as np
import yaml
from PIL import Image
from pathlib import Path

import openslide
from fmc.image_utils import detect_blank_patch

def patch_generator(start, end, size, patch_size, step = None):
    """
    Generates a sequence of patch coordinates.
    """

    if step is None:
        step = patch_size

    width, height = size
    patch_width, patch_height = patch_size

    step_x, step_y = step

    start_x, start_y = start
    end_x, end_y = end

    # print(f'{start=}, {end=}, {size=}, {patch_size=}, {step=}')
    grid_x = 0
    grid_y = 0

    for y in range(start_y, min(end_y, height - patch_height + 1), step_y):
        grid_x = 0
        for x in range(start_x, min(end_x, width - patch_width + 1), step_x):
            yield (x, y, grid_x, grid_y)
            grid_x += 1
        grid_y += 1

class CachcedFileIndex:

    def __init__(self, index_root: Path):
        self.index_root = index_root

    def get_wsi_stuff_path(self, wsi_file_name):
        wsi_fn = Path(wsi_file_name)
        stem = wsi_fn.stem
        suff = wsi_fn.suffix[1:]
        return self.index_root / f'{stem}__{suff}'    

    def get_index_path_s(self, stuff_path, method, x, y) -> Path:
        return stuff_path / method / f'{x}_{y}.png'

    def get_index_path(self, wsi_file_name, method, x, y) -> Path:
        return self.get_index_path_s(self.get_wsi_stuff_path(wsi_file_name), method, x,y)

    def is_calculated(self, wsi_file_name, method, x, y) -> bool:
        return self.get_index_path(wsi_file_name, method, x, y).exists()

    def remove_index(self, wsi_file_name, method, x, y):
        self.get_index_path(wsi_file_name, method, x, y).unlink()

    def get_cached_image(self, wsi_file_name, method, x, y, mode='RGB'):
        return Image.open(self.get_index_path(wsi_file_name, method, x, y)).convert(mode)
    
    def send_to_buffer(self, wsi_file_name, method, x, y):
        file_path = self.get_index_path(wsi_file_name, method, x, y)
        with open(file_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
            buffer.seek(0)
            return buffer


class WSIDriver:
    def get_patch_buffer(self, wsi_file_name, method, x, y, magnification = 0):
        pass

class OpenSlideWSIDriver(WSIDriver):
    def __init__(self, original_files_path, file_index: CachcedFileIndex, patch_size=(224, 224), basic_slide_magnification=2):
        self.file_index = file_index
        self.patch_size = patch_size
        self.original_files_path = original_files_path
        self.basic_slide_magnification = basic_slide_magnification

    def get_patch_buffer(self, wsi_file_name, method, x, y):
        cached_file = self.file_index.get_index_path(wsi_file_name, method, x, y)
        if cached_file.exists():
            # return self.file_index.get_cached_image(wsi_file_name, method, x, y, magnification)
            return self.file_index.send_to_buffer(wsi_file_name, method, x, y)
        else:
            slide = openslide.OpenSlide(str(self.original_files_path / wsi_file_name))
            patch = slide.read_region((x, y), self.basic_slide_magnification, self.patch_size)
            # save to disk cache
            patch.save(cached_file)
            # send to buffer
            buff = io.BytesIO()
            patch.save(buff, format='PNG')
            buff.seek(0)
            return buff

    def save_patch_from_wsi(self, wsi_file_name, x, y):
        method = 'original'
        cached_file = self.file_index.get_index_path(wsi_file_name, method, x, y)
        cached_file.parent.mkdir(parents=True, exist_ok=True)

        if cached_file.exists():
            # return self.file_index.get_cached_image(wsi_file_name, method, x, y, magnification)
            return self.file_index.send_to_buffer(wsi_file_name, method, x, y)
        else:
            slide = openslide.OpenSlide(str(self.original_files_path / wsi_file_name))
            patch = slide.read_region((x, y), self.basic_slide_magnification, self.patch_size)
            # save to disk cache
            patch.save(cached_file)


    def save_metadata(self, wsi_file_name):
        slide = openslide.OpenSlide(str(self.original_files_path / wsi_file_name))  
        stuff_path = self.file_index.get_wsi_stuff_path(wsi_file_name)
        stuff_path.mkdir(parents=True, exist_ok=True)
        with open(stuff_path / 'metadata.yaml', 'w') as yaml_file:
            d = {k: v for k,v in slide.properties.items()}
            yaml.dump(d, yaml_file)      

    def generate_patches(self, wsi_file_name, start=(0, 0), end=None):
        slide = openslide.OpenSlide(str(self.original_files_path / wsi_file_name))
        width, height = slide.level_dimensions[0]

        if end is None:
            end = (width, height)

        (self.file_index.get_wsi_stuff_path(wsi_file_name) / 'original').mkdir(exist_ok=True)
        threshold = 0. # 0 -> не отсеивать
        cc = 0
        cc_d = 0
        total = 0

        # read_region читает в координатах 0 уровня WSI, но возвращает изображение размера patch_size для указанного magnification
        step_x = self.patch_size[0] * self.basic_slide_magnification * 2
        step_y = self.patch_size[1] * self.basic_slide_magnification * 2
        print(f'{width=}, {height=}, cols: {width//step_x} rows: {height//step_y} {step_x=}, {step_y=} {self.basic_slide_magnification=}')

        # start = (30000+1000*3, 55000)
        # end=(width, height)
        # end=(start[0]+3000, start[1]+3000)

        for r in patch_generator(start=start, end=end, size=(width, height), patch_size=self.patch_size, step = (step_x, step_y)):
            patch = slide.read_region(r[:2], self.basic_slide_magnification, self.patch_size)

            _, _, is_blank = detect_blank_patch(np.array(patch), threshold=threshold)
            if not is_blank:
                patch.save(self.file_index.get_index_path(wsi_file_name, 'original', r[2]*self.patch_size[0], r[3]*self.patch_size[1]))
                cc += 1
                # print(f'{r}')
            else:
                cc_d += 1
            total+=1

            if total%1000 == 0:
                print(f'Total: {total}, Saved: {cc}, blank: {cc_d}')
        print(f'Total: {total}, Saved: {cc}, blank: {cc_d}')


class FeatureMapWSIDriver(WSIDriver):
    def __init__(self, original_files_path, file_index: CachcedFileIndex, patch_size=(224, 224)):
        pass


