import argparse
import shutil
from pathlib import Path

import random

import json
from tqdm import tqdm

from terial import controllers
from terial.database import session_scope


parser = argparse.ArgumentParser()
parser.add_argument(dest='in_path', type=Path)
parser.add_argument(dest='out_dir', type=Path)
args = parser.parse_args()


def main():
    tqdm.write(f'Loading pairs')

    base_dir = Path('/projects/grail/kparnb/data/terial/brdf-classifier/inference')
    pop_names = ['photo', 'default', 'random', 'flagship']
    pop_addrs = [
        ('default', 'default'),
        ('random', 'random'),
        ('flagship/45', 'flagship'),
    ]

    with open(args.in_path, 'r') as f:
        pair_ids = json.load(f)

    for pair_id in tqdm(pair_ids):
        for pop_dir, pop_name in pop_addrs:
            src_path = base_dir / pop_dir / f'{pair_id}.json'
            dst_dir = args.out_dir / pop_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / f'{pair_id}.json'
            shutil.copy(src_path, dst_path)


if __name__ == '__main__':
    main()