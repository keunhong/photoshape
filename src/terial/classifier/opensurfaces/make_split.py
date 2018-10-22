import argparse
from pathlib import Path

import csv

import random

import json

from terial.classifier import opensurfaces

parser = argparse.ArgumentParser()
parser.add_argument(dest='base_dir', type=Path,
                    default='/local1/kpar/data/opensurfaces')
parser.add_argument(dest='out_path', type=Path)
args = parser.parse_args()


def main():
    subst_name_to_id = {}
    with (args.base_dir / 'label-substance-colors.csv').open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subst_name_to_id[row['substance_name']] = row['substance_id']

    photo_ids = set()
    with (args.base_dir / 'shapes.csv').open('r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subst_id = subst_name_to_id.get(row['substance_name'])
            if subst_id is None:
                continue
            subst_id = int(subst_id)
            if subst_id not in opensurfaces.OSURF_SUBST_ID_TO_SUBST:
                continue

            photo_id = int(row['photo_id'])
            shape_id = int(row['shape_id'])
            photo_ids.add(photo_id)

    photo_ids = list(photo_ids)
    num_train = int(len(photo_ids) * 0.9)

    random.shuffle(photo_ids)

    train_photo_ids = photo_ids[:num_train]
    validation_photo_ids = photo_ids[num_train:]

    split = {
        'train': train_photo_ids,
        'validation': validation_photo_ids,
    }

    with args.out_path.open('w')as f:
        json.dump(split, f, indent=2)


if __name__ == '__main__':
    main()
