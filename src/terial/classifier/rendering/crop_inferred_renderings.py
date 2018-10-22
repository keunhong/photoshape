import argparse
import multiprocessing
import time
from functools import partial
from pathlib import Path

from scipy.misc import imresize
from skimage.io import imread, imsave
from tqdm import tqdm

from terial import models, config, controllers
from terial.database import session_scope
from toolbox.images import mask_bbox, bbox_make_square, crop_bbox

parser = argparse.ArgumentParser()
parser.add_argument(dest='input_dir', type=Path)
parser.add_argument(dest='output_dir', type=Path)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()


def worker(input_tup):
    pair, input_path = input_tup
    output_path = args.output_dir / f"{input_path.stem}.jpg"
    output_sm_path = args.output_dir / f"{input_path.stem}.small.jpg"
    if output_path.exists() and not args.overwrite:
        return

    try:
        rendering = imread(input_path)
    except OSError:
        tqdm.write(f"Cannot read {input_path!s}")
        input_path.unlink()
        return
    fg_mask = pair.load_data(config.PAIR_FG_BBOX_NAME)
    rendering = imresize(rendering, fg_mask.shape)
    bbox = bbox_make_square(mask_bbox(fg_mask))
    cropped = crop_bbox(rendering, bbox)
    cropped = imresize(cropped, (750, 750))
    imsave(output_path, cropped)
    cropped = imresize(cropped, (100, 100))
    imsave(output_sm_path, cropped)


def main():
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = list(args.input_dir.glob('*.jpg'))
    print("Fetching pairs.")
    with session_scope() as sess:
        pairs, _ = controllers.fetch_pairs_default(sess)
        pairs_by_id = {p.id: p for p in pairs}

    input_tups = []
    for input_path in tqdm(input_paths):
        pair_id = int(input_path.name.split('.')[0])
        if pair_id not in pairs_by_id:
            continue
        pair = pairs_by_id[pair_id]
        input_tups.append((pair, input_path))

    pbar = tqdm(total=len(input_tups))
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for i in pool.imap_unordered(worker, input_tups):
            pbar.update(1)




if __name__ == '__main__':
    main()

