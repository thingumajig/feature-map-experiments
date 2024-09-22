import io
from pathlib import Path

from PIL import Image



class CachcedFileIndex:

    def __init__(self, index_root: Path):
        self.index_root = index_root

    def get_index_path(self, wsi_file_name, method, x, y, magnification = 0) -> Path:
        wsi_fn = Path(wsi_file_name)
        stem = wsi_fn.stem
        suff = wsi_fn.suffix[1:]
        return self.index_root / f'{stem}__{suff}' / method / f'{x}_{y}_{magnification}.png'

    def is_calculated(self, wsi_file_name, method, x, y, magnification = 0) -> bool:
        return self.get_index_path(wsi_file_name, method, x, y, magnification).exists()

    def remove_index(self, wsi_file_name, method, x, y, magnification = 0):
        self.get_index_path(wsi_file_name, method, x, y, magnification).unlink()

    def get_cached_image(self, wsi_file_name, method, x, y, magnification = 0, mode='RGB'):
        return Image.open(self.get_index_path(wsi_file_name, method, x, y, magnification)).convert(mode)
    
    def send_to_buffer(self, wsi_file_name, method, x, y, magnification = 0):
        file_path = self.get_index_path(wsi_file_name, method, x, y, magnification)
        with open(file_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
            buffer.seek(0)
            return buffer


import openslide


class WSIDriver:
    def get_patch_buffer(self, wsi_file_name, method, x, y, magnification = 0):
        pass

class OpenSlideWSIDriver(WSIDriver):
    def __init__(self, original_files_path, file_index: CachcedFileIndex, patch_size=(224, 224)):
        self.file_index = file_index
        self.patch_size = patch_size
        self.original_files_path = original_files_path


    def get_patch_buffer(self, wsi_file_name, method, x, y, magnification = 0):
        cached_file = self.file_index.get_index_path(wsi_file_name, method, x, y, magnification)
        if cached_file.exists():
            # return self.file_index.get_cached_image(wsi_file_name, method, x, y, magnification)
            return self.file_index.send_to_buffer(wsi_file_name, method, x, y, magnification)
        else:
            slide = openslide.OpenSlide(str(self.original_files_path / wsi_file_name))
            patch = slide.read_region((x, y), magnification, self.patch_size)
            # save to disk cache
            patch.save(cached_file)
            # send to buffer
            buff = io.BytesIO()
            patch.save(buff, format='PNG')
            buff.seek(0)
            return buff
        

class FeatureMapWSIDriver(WSIDriver):
    def __init__(self, original_files_path, file_index: CachcedFileIndex, patch_size=(224, 224)):
        pass


