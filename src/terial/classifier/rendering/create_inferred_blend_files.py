"""
Computes aligning features for all ShapeNet shapes.
"""
import argparse
import multiprocessing
import subprocess
from functools import partial
from pathlib import Path

import visdom
import warnings

from tqdm import tqdm

from terial import controllers
from terial.database import session_scope
from terial.models import ExemplarShapePair

parser = argparse.ArgumentParser()
parser.add_argument(dest='inference_dir', type=Path)
parser.add_argument(dest='out_name', type=Path)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--output-mtl', action='store_true')
parser.add_argument('--mtl-only', action='store_true')
parser.add_argument('--use-minc-substances', action='store_true')
parser.add_argument('--frontal', action='store_true')
parser.add_argument('--diagonal', action='store_true')
parser.add_argument('--no-floor', action='store_true')
parser.add_argument('--category', type=str)
args = parser.parse_args()


scene_types = {'inferred'}
if args.output_mtl:
    scene_types.add('mtl')

if args.mtl_only:
    scene_types = {'mtl'}


def worker(pair_id, renderings_dir):
    inference_path = args.inference_dir / f'{pair_id}.json'
    if not inference_path.exists():
        tqdm.write(f'{inference_path!s} does not exist')
        return

    for scene_type in scene_types:
        out_path = (
                renderings_dir / f'{inference_path.stem}.{scene_type}.blend')
        # if out_path.exists():
        #     continue

        command = [
            'python', '-m', 'terial.classifier.rendering.create_blend_file',
            str(inference_path),
            str(out_path),
            '--type', scene_type,
            # '--use-weighted-scores',
            '--pack-assets',
        ]
        if args.frontal:
            command.append('--frontal')
        elif args.diagonal:
            command.append('--diagonal')
        if args.use_minc_substances:
            command.append('--use-minc-substances')
        if args.no_floor:
            command.append('--no-floor')
        print(f' * Launching command {command}')
        subprocess.call(command)
    return pair_id


def main():
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')
    # if args.frontal:
    #     renderings_dir = args.inference_dir / 'blend-frontal'
    # elif args.frontal:
    #     renderings_dir = args.inference_dir / 'blend-diagonal'
    # else:
    #     renderings_dir = args.inference_dir / 'blend'
    renderings_dir = args.inference_dir / args.out_name
    renderings_dir.mkdir(parents=True, exist_ok=True)

    filters = []
    if args.category is not None:
        filters.append(ExemplarShapePair.shape.has(category=args.category))

    with session_scope() as sess:
        # pairs, count = controllers.fetch_pairs_default(sess)
        pairs, count = controllers.fetch_pairs(
            sess,
            filters=filters,
            by_shape=False,
            order_by=ExemplarShapePair.id.asc(),
        )

    pool = multiprocessing.Pool(processes=args.num_workers)

    pair_ids = [p.id for p in pairs]

    pbar = tqdm(total=len(pairs))
    for i in pool.imap_unordered(partial(worker, renderings_dir=renderings_dir),
                                 pair_ids):
        pbar.set_description(str(i))
        pbar.update(1)


if __name__ == '__main__':
    main()
