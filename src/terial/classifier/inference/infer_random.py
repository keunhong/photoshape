import argparse
import multiprocessing
import ujson
from collections import defaultdict
from functools import partial
from pathlib import Path

import random
from tqdm import tqdm

from terial import controllers, config, models
from terial.config import SUBSTANCES
from terial.database import session_scope
from terial.pairs.utils import compute_segment_substances

parser = argparse.ArgumentParser()
parser.add_argument(dest='out_dir', type=Path)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--num-workers', type=int, default=4)
args = parser.parse_args()


def worker(pair, mat_id_by_subst):
    out_path = Path(args.out_dir, f'{pair.id}.json')
    if out_path.exists():
        return

    try:
        seg_substances = compute_segment_substances(pair, return_ids=True)
    except FileNotFoundError:
        tqdm.write(f"Pair {pair.id} does not have segment map")
        return

    result_dict = {
        'pair_id': pair.id,
        'segments': {},
    }

    for seg_id, subst_id in seg_substances.items():
        subst_name = SUBSTANCES[subst_id]
        if subst_name == 'background':
            continue
        # Generate 10 random ones just in case.
        result_dict['segments'][str(seg_id)] =  {
            'material': [{
                'score': 1,
                'id': random.choice(mat_id_by_subst[subst_name]),
            } for _ in range(10)]
        }

    with out_path.open('w') as f:
        ujson.dump(result_dict, f, indent=2)


def main():
    out_dir = args.out_dir
    out_dir.mkdir(exist_ok=True, parents=True)


    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs_default(sess)
        materials = sess.query(models.Material).all()
        mat_by_id = {m.id: m for m in materials}

    pairs = [
        pair for pair in pairs
        if args.overwrite or not (Path(out_dir, f'{pair.id}.json').exists())
    ]
    print(len(pairs))

    # pairs = []
    mat_id_by_subst = defaultdict(list)

    tqdm.write(f"Fetching shapes and their pairs.")
    with session_scope() as sess:
        materials = sess.query(models.Material).filter_by(enabled=True).all()
        for material in materials:
            mat_id_by_subst[material.substance].append(material.id)

        # shapes, _ = controllers.fetch_shapes_with_pairs(sess)
        # for shape in tqdm(shapes):
        #     _pairs = shape.get_topk_pairs(config.INFERENCE_TOPK,
        #                                  config.INFERENCE_MAX_DIST)
        #     for pair in _pairs:
        #         pair.exemplar, pair.shape
        #     pairs.extend(_pairs)

    pool = multiprocessing.Pool(processes=args.num_workers)

    tqdm.write(f"Processing {len(pairs)} pairs")
    pbar = tqdm(total=len(pairs))
    for i in pool.imap_unordered(
            partial(worker, mat_id_by_subst=mat_id_by_subst), pairs):
        pbar.update(1)


if __name__ == '__main__':
    main()
