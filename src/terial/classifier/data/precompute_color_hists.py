import io
import ujson
from collections import defaultdict

import lmdb
import argparse

from pathlib import Path

import numpy as np
import msgpack
from tqdm import tqdm
from multiprocessing import Pool

from terial.classifier.data.utils import bytes_to_image
from terial.classifier.utils import ColorBinner

parser = argparse.ArgumentParser()
parser.add_argument(dest='lmdb_path', type=Path)
parser.add_argument('--space', type=str, default='lab')
parser.add_argument('--shape', type=str, default='3,5,5')
parser.add_argument('--sigma', type=str, default='0.5,1.0,1.0')
parser.add_argument('--num-workers', default=4, type=int)
args = parser.parse_args()


color_binner = ColorBinner(
    space=args.space,
    shape=tuple(int(i) for i in args.shape.split(',')),
    sigma=tuple(float(i) for i in args.sigma.split(',')),
)


env = None
def worker(key):
    global env
    if env is None:
        env = lmdb.open(str(args.lmdb_path),
                        readonly=True,
                        max_readers=1,
                        lock=False,
                        readahead=False,
                        meminit=False)

    color_hists = defaultdict(dict)

    image_buf = io.BytesIO()
    mask_buf = io.BytesIO()

    with env.begin(write=False) as txn:
        payload = msgpack.unpackb(txn.get(key))
        ldr_bytes = payload[b'ldr_image']
        image = bytes_to_image(ldr_bytes, buf=image_buf)
        seg_map_bytes = payload[b'segment_map']
        seg_map = np.array(bytes_to_image(seg_map_bytes, buf=mask_buf),
                           dtype=int) - 1

        for seg_id in payload[b'seg_material_ids'].keys():
            mask = (seg_map == seg_id)
            hist = color_binner.compute(image, mask)

            color_hists[key.decode()][int(seg_id)] = hist.tolist()

    return color_hists


def main():

    env = lmdb.open(str(args.lmdb_path),
                    readonly=True,
                    max_readers=1,
                    lock=False,
                    readahead=False,
                    meminit=False)

    out_dir = args.lmdb_path / 'color-hists'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{color_binner.name}.msgpack'

    with env.begin(write=False) as txn:
        keys = [key for key, _ in txn.cursor()]
        length = txn.stat()['entries']

    pbar = tqdm(total=length)

    color_hists = {}

    with Pool(processes=args.num_workers) as pool:
        for h in pool.imap_unordered(worker, keys):
            color_hists.update(h)
            pbar.update(1)

    print(f"Saving to {out_path!s}")

    with out_path.open('wb') as f:
        msgpack.pack(color_hists, f)


if __name__ == '__main__':
    main()
