import io
from pathlib import Path

import lmdb

import msgpack
import random
import numpy as np
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from terial.classifier.utils import ColorBinner
from terial.config import SUBSTANCES
from terial.classifier.data.utils import bytes_to_image


class MaterialRendDataset(torch.utils.data.Dataset):

    @classmethod
    def build_flattened(cls, env, keys):
        key_seg_pairs = []
        with env.begin(write=False) as tx:
            for key in tqdm(keys):
                payload = msgpack.unpackb(tx.get(key))
                for seg_id in payload[b'seg_material_ids'].keys():
                    key_seg_pairs.append((key, seg_id, payload))

        return key_seg_pairs

    def __init__(self,
                 lmdb_path,
                 meta_dict,
                 shape, image_transform, mask_transform,
                 lmdb_name,
                 color_binner: ColorBinner =None,
                 mask_noise_p=0.0,
                 ):
        self.lmdb_name = lmdb_name
        self.env = lmdb.open(str(lmdb_path),
                             readonly=True,
                             max_readers=1,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.keys = [key for key, _ in txn.cursor()]
            self.length = txn.stat()['entries']

        print(f" * Pre-loading data from lmdb.")
        self.key_seg_pairs = self.build_flattened(self.env, self.keys)

        self.mat_id_to_label = {
            int(k): int(v) for k, v in meta_dict['mat_id_to_label'].items()
        }
        # self.mat_id_to_color_hist = mat_id_to_color_hist
        self.shape = shape
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.sysrand = random.SystemRandom()
        self.mask_noise_p = mask_noise_p

        self.color_binner = color_binner
        if color_binner:
            color_hist_path = (lmdb_path / 'color-hists'
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

        self.image_buf = io.BytesIO()
        self.mask_buf = io.BytesIO()

    def __len__(self):
        return len(self.key_seg_pairs)

    def __getitem__(self, idx):
        key, seg_id, payload = self.key_seg_pairs[idx]

        ldr_bytes = payload[b'ldr_image']
        image = bytes_to_image(ldr_bytes, buf=self.image_buf)
        seg_map_bytes = payload[b'segment_map']
        seg_map = np.array(bytes_to_image(seg_map_bytes, buf=self.mask_buf),
                           dtype=int) - 1

        mask = (seg_map == seg_id)

        mask = mask[:, :, np.newaxis].astype(dtype=np.uint8) * 255

        mat_id = payload[b'seg_material_ids'][seg_id]
        subst_name = payload[b'seg_substances'][seg_id].decode()

        # Re-seed to ensure same transforms are applied.
        # System random must be used so that we don't seed base on time.
        seed = self.sysrand.randint(0, 2147483647)
        random.seed(seed)
        np.random.seed(seed)
        transformed_image = self.image_transform(image.copy())
        random.seed(seed)
        np.random.seed(seed)
        mask = self.mask_transform(mask)
        mask_noise_pos = torch.rand(*mask.size()) < self.mask_noise_p
        mask_noise_neg = torch.rand(*mask.size()) < self.mask_noise_p
        mask[mask_noise_pos] = 1.0
        mask[mask_noise_neg] = 0.0

        image_tensor = torch.cat((transformed_image, mask), dim=0)
        material_label = self.mat_id_to_label[mat_id]
        substance_label = SUBSTANCES.index(subst_name)

        retval = {
            'image': image_tensor,
            'material_label': material_label,
            'substance_label': substance_label,
        }

        if self.color_binner and self.color_hist_dict:
            retval['color_hist'] = \
                torch.FloatTensor(self.color_hist_dict[key][seg_id])
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
