import argparse
import multiprocessing
import shutil
from pathlib import Path

import numpy as np
import visdom
from skimage.io import imread
from tqdm import tqdm

vis = visdom.Visdom(env='fix-purple', port=8097)


parser = argparse.ArgumentParser()
parser.add_argument(dest='rend_path', type=Path)
parser.add_argument('--num-workers', type=int, default=4)
args = parser.parse_args()


def worker(path):
    try:
        image = imread(path)
    except OSError:
        print('nope', str(path))
        return
    purple_mask = ((image[:, :, 0] > 200)
                   & (image[:, :, 1] < 60)
                   & (image[:, :, 2] > 200))
    if purple_mask.sum() > 50:
        print(str(path))
        print(str(args.rend_path.parent / 'renderings' / f'{path.stem}.png'))

        vis.image(image.transpose((2, 0, 1)), win='image')
        vis.image(purple_mask.astype(np.uint8) * 255, win='mask')
        path.unlink()


def main():
    paths = list(args.rend_path.glob('*.jpg'))
    pool = multiprocessing.Pool()
    for i in pool.imap_unordered(worker, paths):
        pass


if __name__ == '__main__':
    main()

