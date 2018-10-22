import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import msgpack
import warnings
from skimage.io import imread
from tqdm import tqdm

from terial.classifier.opensurfaces.dataset import read_shape_dicts
from terial.classifier.utils import ColorBinner


parser = argparse.ArgumentParser()
parser.add_argument(dest='base_dir', type=Path)
parser.add_argument('--space', choices=['rgb', 'lab'], required=True)
parser.add_argument('--shape', default='3,5,5', required=True)
parser.add_argument('--sigma', default='0.5,1,1', required=True)
parser.add_argument('--num-workers', type=int, default=4)
args = parser.parse_args()


color_binner = ColorBinner(
    space=args.space,
    shape=tuple(int(i) for i in args.shape.split(',')),
    sigma=tuple(float(i) for i in args.sigma.split(',')),
)


def compute_colorhist(shape_dict, base_dir):
    warnings.filterwarnings("error")
    photo_id = shape_dict['photo_id']
    shape_id = shape_dict['shape_id']
    photo_path = base_dir / 'shapes-cropped' / f'{shape_id}.png'

    if not photo_path.exists():
        return None, None

    try:
        photo_im = imread(photo_path)
    except (OSError, ValueError):
        # photo_path.unlink()
        print(f"{photo_path!s} corrupt")
        return None, None

    hist = color_binner.compute(photo_im[:, :, :3], photo_im[:, :, 3] > 0)

    return shape_id, hist.tolist()


def main():
    photos_dir = args.base_dir / 'photos'
    out_dir = args.base_dir / 'shape-color-hists'
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f'{color_binner.name}.msgpack'

    shape_dicts = read_shape_dicts(args.base_dir / 'shapes.csv')

    pbar = tqdm(shape_dicts)

    shape_hists = {}

    pool = Pool(processes=args.num_workers)
    for shape_id, hist in pool.imap_unordered(
            partial(compute_colorhist, base_dir=args.base_dir), shape_dicts):
        if shape_id is not None:
            shape_hists[shape_id] = hist
        pbar.update(1)

    print(f"Saving to {out_path!s}")

    with out_path.open('wb') as f:
        msgpack.pack(shape_hists, f)


if __name__ == '__main__':
    main()

