import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(dest='base_dir', type=Path)
parser.add_argument(dest='out_dir', type=Path)
args = parser.parse_args()


def resize_and_save(photo_path, base_dir, out_dir):
    photo_id = photo_path.stem
    out_path = args.out_dir / photo_path.name
    if out_path.exists():
        return
    label_path = base_dir / 'photos-labels' / f'{photo_id}.png'
    photo_im = Image.open(str(photo_path))
    label_im = Image.open(str(label_path))
    photo_im = photo_im.resize(label_im.size)
    photo_im.save(out_path)


def main():
    photos_dir = args.base_dir / 'photos'
    args.out_dir.mkdir(exist_ok=True, parents=True)

    photo_paths = list(photos_dir.glob('*.jpg'))
    pbar = tqdm(photo_paths)

    pool = Pool(processes=4)
    for i in pool.imap_unordered(partial(resize_and_save,
                                         base_dir=args.base_dir,
                                         out_dir=args.out_dir),
                                 photo_paths):
        pbar.update(1)


if __name__ == '__main__':
    main()

