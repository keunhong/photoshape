import argparse
from pathlib import Path

import csv
import random

import json

from sqlalchemy import orm
from tqdm import tqdm

from terial import models
from terial.database import session_scope


parser = argparse.ArgumentParser()
parser.add_argument(dest='out_dir', type=Path)
args = parser.parse_args()


def main():
    tqdm.write(f'Loading pairs')

    base_dir = Path('/projects/grail/kparnb/data/terial/turk-study/inference')
    pop_names = ['photos', 'default', 'random', 'flagship']
    pop_addrs = [
        ('default/rends-nofloor-cropped', 'mtl', 'default'),
        ('random/rends-nofloor-cropped', 'inferred', 'random'),
        ('flagship/rends-nofloor-cropped', 'inferred', 'flagship'),
    ]

    with (base_dir / 'hermanmiller_pairs.json').open('r') as f:
        hermanmiller_pair_ids = json.load(f)

    with (base_dir / 'shapenet_pairs.json').open('r') as f:
        shapenet_pair_ids = json.load(f)

    with session_scope() as sess:
        hermanmiller_pairs = (
            sess.query(models.ExemplarShapePair)
                .options(orm.joinedload(models.ExemplarShapePair.exemplar))
                .filter(models.ExemplarShapePair.id.in_(hermanmiller_pair_ids))
                .all())
        shapenet_pairs = (
            sess.query(models.ExemplarShapePair)
                .options(orm.joinedload(models.ExemplarShapePair.exemplar))
                .filter(models.ExemplarShapePair.id.in_(shapenet_pair_ids))
                .all())

    datasets = [
        ('shapenet', shapenet_pairs),
        ('hermanmiller', hermanmiller_pairs),
    ]

    images_dir = args.out_dir / 'images'

    tqdm.write(f'Saving pair photos')
    exemplar_out_dir = images_dir / 'photos'
    exemplar_out_dir.mkdir(exist_ok=True, parents=True)

    fieldnames = ['pair_id', 'dset_name', 'pop_name', 'url']

    for dset_name, dset_samps in datasets:
        rows = []
        for pop_name in pop_names:
            for pair in dset_samps:
                rows.append({
                    'dset_name': dset_name,
                    'pop_name': pop_name,
                    'pair_id': pair.id,
                    'url': f'{dset_name}/{pop_name}/{pair.id}.jpg' if pop_name != 'photos' else f'photos/{pair.id}.jpg'
                })

        random.shuffle(rows)

        csv_path = args.out_dir / f'{dset_name}.csv'
        with csv_path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)



if __name__ == '__main__':
    main()
