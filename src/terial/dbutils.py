import os
import enum

import numpy as np
import skimage.io

import toolbox.images
import toolbox.io
import toolbox.io.images


class SerializeMixin(object):

    @classmethod
    def serialize_type(cls, value):
        if isinstance(value, enum.Enum):
            return value.name
        elif isinstance(value, dict):
            return {k: cls.serialize_type(v) for k, v in value.items()}
        return value

    def serialize(self):
        d = {
            k: self.serialize_type(v) for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        return d


class BlobDataMixin(object):

    @property
    def data_path(self):
        raise NotImplementedError

    def get_data_path(self, name):
        _, ext = os.path.splitext(name)
        if ext in {'.png', '.jpg'}:
            suffix = '.png'
            return self.get_image_dir_path() / f'{name.replace(suffix, "")}{suffix}'
        elif ext in {'.hdr', '.exr'}:
            suffix = '.exr'
            return self.get_image_dir_path() / f'{name.replace(suffix, "")}{suffix}'
        elif ext in {'.npz', '.npy'}:
            suffix = '.npz'
            return self.get_numpy_dir_path() / f'{name.replace(suffix, "")}{suffix}'
        else:
            raise ValueError(f'Unsupported extension {ext!r}')

    def get_numpy_data_path(self, name):
        name = name.rstrip('.npz')
        return self.get_numpy_dir_path() / f'{name}.npz'

    def get_image_data_path(self, name):
        name = name.rstrip('.png')
        return self.get_image_dir_path() / f'{name}.png'

    def get_numpy_dir_path(self):
        return self.data_path / 'numpy'

    def get_image_dir_path(self):
        return self.data_path / 'images'

    def save_data(self, name, data):
        _, ext = os.path.splitext(name)
        if ext in {'.png', '.jpg'}:
            path = self.get_data_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if (data.dtype == np.float32
                    or data.dtype == np.float64
                    or data.dtype == float):
                data = (data * 255).astype(np.uint8)
            skimage.io.imsave(str(path), data)
        elif ext in {'.hdr', '.exr'}:
            path = self.get_data_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            toolbox.io.images.save_hdr(path, data)
        elif ext in {'.npz', '.npy'}:
            path = self.get_data_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                np.savez(f, data)
        else:
            raise ValueError(f'Unsupported extension {ext!r}')

    def load_data(self, name):
        _, ext = os.path.splitext(name)
        if ext in {'.png', '.jpg'}:
            path = self.get_data_path(name)
            return _load_image(path)
        elif ext in {'.exr', '.hdr'}:
            path = self.get_data_path(name)
            toolbox.io.images.load_hdr(path)
        elif ext in {'.npz', '.npy'}:
            path = self.get_data_path(name)
            if not path.exists():
                raise FileNotFoundError(f'Data {path!s} not found.')
            with open(path, 'rb') as f:
                return np.load(f)['arr_0']
        else:
            raise ValueError(f'Unsupported type {ext!r}')

    def data_exists(self, name, type=None):
        return self.get_data_path(name).exists()


def _load_image(path):
    image = skimage.io.imread(str(path))
    if '.map.' in path.name:
        return image.astype(int)
    return skimage.img_as_float32(image)
