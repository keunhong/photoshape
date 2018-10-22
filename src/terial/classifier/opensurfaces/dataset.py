import csv

import random
import typing
from functools import partial
from pathlib import Path

import json

import msgpack
import numpy as np
import torch.utils.data
from PIL import Image, ImageFile
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

import toolbox.images
from terial.classifier import opensurfaces as osurf
from terial.config import SUBSTANCES
from toolbox.colors import compute_lab_histogram


ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_shape_dicts(csv_path: Path, photo_ids=None):
    print(f"Reading shapes from {csv_path!s}")
    shape_dicts = []
    with csv_path.open('r') as f:
        reader = list(csv.DictReader(f))
        for row in tqdm(reader):
            photo_id = int(row['photo_id'])
            shape_id = int(row['shape_id'])
            if photo_ids and photo_id not in photo_ids:
                continue
            substance_id = osurf.OSURF_SUBST_NAME_TO_ID.get(
                row['substance_name'])
            if substance_id is None:
                continue
            substance_code = osurf.OSURF_SUBST_ID_TO_COLOR[substance_id]
            substance = osurf.OSURF_SUBST_ID_TO_SUBST.get(substance_id)
            if substance is None:
                continue

            label_code = osurf.OSURF_LABEL_NAME_TO_COLOR.get(row['name_name'], 0)
            try:
                base_color = (int(255 * float(row['albedo_r'])),
                              int(255 * float(row['albedo_g'])),
                              int(255 * float(row['albedo_b'])))
            except ValueError:
                base_color = None

            shape_dicts.append({
                'photo_id': photo_id,
                'shape_id': shape_id,
                'substance_code': substance_code,
                'label_code': label_code,
                'substance': substance,
                'base_color': base_color,
            })
    return shape_dicts


def filter_nonexistent(base_dir, shape_dicts):
    return [s for s in shape_dicts
            if (base_dir / 'photos-resized' / f'{s["photo_id"]}.jpg').exists()]



class OpenSurfacesDataset(torch.utils.data.Dataset):

    def __init__(self,
                 base_dir: Path,
                 photo_ids: typing.List[int],
                 image_transform,
                 mask_transform,
                 cropped_image_transform,
                 cropped_mask_transform,
                 color_binner=None,
                 use_cropped=False,
                 p_cropped=0.5):
        self.base_dir = base_dir
        self.photo_ids = set(photo_ids)
        self.use_cropped = use_cropped
        self.p_cropped = p_cropped

        self.color_binner = color_binner
        if color_binner:
            color_hist_path = (base_dir / 'shape-color-hists'
                               / f'{color_binner.name}.msgpack')
            if color_hist_path.exists():
                print(f" * Loading color histogram from {color_hist_path!s}")
                with color_hist_path.open('rb') as f:
                    self.color_hist_dict = msgpack.unpack(f)
            else:
                print(f" * {color_hist_path!s} does not exist.")
                print(f" * WARNING! Color hist not precomputed, this will be slow!")
                self.color_hist_dict = None
        else:
            self.color_hist_dict = None

        self.shape_dicts = []
        print(f"Loading shapes.")
        self.shape_dicts = filter_nonexistent(
            base_dir,
            read_shape_dicts(base_dir / 'shapes.csv', photo_ids))

        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.cropped_image_transform = cropped_image_transform
        self.cropped_mask_transform = cropped_mask_transform
        self.sysrand = random.SystemRandom()

    def __len__(self):
        return len(self.shape_dicts)

    def __getitem__(self, idx):
        padding = 224
        shape_dict = self.shape_dicts[idx]
        photo_id = shape_dict['photo_id']
        shape_id = shape_dict['shape_id']

        use_cropped = (self.use_cropped
                       and self.sysrand.random() <= self.p_cropped)

        if use_cropped:
            image_transform = self.cropped_image_transform
            mask_transform = self.cropped_mask_transform

            photo_path = self.base_dir / 'shapes-cropped' / f'{shape_id}.png'
            photo_im = Image.open(str(photo_path))
            photo_arr = np.array(photo_im)
            label_mask = photo_arr[:, :, 3] > 0
            label_mask = label_mask[:, :, np.newaxis].astype(np.uint8) * 255
        else:
            image_transform = self.image_transform
            mask_transform = self.mask_transform

            photo_im = Image.open(str(self.base_dir / 'photos-resized' / f'{photo_id}.jpg'))
            label_im = Image.open(self.base_dir / 'photos-labels' / f'{photo_id}.png')
            label_im = np.array(label_im, dtype=np.uint8)
            height, width = label_im.shape[:2]
            photo_im = photo_im.resize((width, height))
            label_mask = ((label_im[:, :, 0] == shape_dict['substance_code'])
                          & (label_im[:, :, 1] == shape_dict['label_code']))

            if label_mask.sum() > 0:
                mask_bbox = toolbox.images.mask_bbox(label_mask)
                mask_bbox = toolbox.images.clamp_bbox(
                    (mask_bbox[0] - padding, mask_bbox[1] + padding,
                     mask_bbox[2] - padding, mask_bbox[3] + padding),
                    (height, width))
            else:
                mask_bbox = (0, height, 0, width)
            label_mask = label_mask[:, :, np.newaxis].astype(np.uint8) * 255
            photo_im = toolbox.images.crop_bbox(photo_im, mask_bbox)
            label_mask = toolbox.images.crop_bbox(label_mask, mask_bbox)

        # Re-seed to ensure same transforms are applied.
        # System random must be used so that we don't seed base on time.
        seed = self.sysrand.randint(0, 2147483647)
        random.seed(seed)
        np.random.seed(seed)
        transformed_image = image_transform(photo_im.copy().convert('RGB'))
        random.seed(seed)
        np.random.seed(seed)
        mask = mask_transform(label_mask)

        image_tensor = torch.cat((transformed_image, mask), dim=0)
        substance_label = SUBSTANCES.index(shape_dict['substance'])

        retval = {
            'image': image_tensor,
            'substance_label': substance_label,
        }

        if self.color_binner and self.color_hist_dict:
            retval['color_hist'] = \
                torch.FloatTensor(self.color_hist_dict[shape_id])
        elif self.color_binner:
            raise NotImplementedError

        return retval


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    flattened_batch = []
    for data in batch:
        num_examples = len(data['image'])
        for i in range(num_examples):
            flattened_batch.append({
                k: v[i] for k, v in data.items()
            })

    return default_collate(flattened_batch)
