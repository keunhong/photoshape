import argparse
from pathlib import Path

import random

import json
from tqdm import tqdm

from terial import controllers
from terial.database import session_scope


parser = argparse.ArgumentParser()
parser.add_argument(dest='out_dir', type=Path)
args = parser.parse_args()


def main():
    tqdm.write(f'Loading pairs')

    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs_default(sess)

    sn_pairs = []
    hm_pairs = []

    for pair in tqdm(pairs):
        if pair.shape.source == 'shapenet':
            sn_pairs.append(pair)
        elif pair.shape.source == 'hermanmiller':
            hm_pairs.append(pair)

    turk_inf_dir = args.out_dir
    turk_inf_dir.mkdir(parents=True, exist_ok=True)

    sn_samples = random.sample(sn_pairs, 1000)
    hm_samples = random.sample(hm_pairs, 500)

    with open(turk_inf_dir / 'shapenet_pairs.json', 'w') as f:
        json.dump([p.id for p in sn_samples], f)

    with open(turk_inf_dir / 'hermanmiller_pairs.json', 'w') as f:
        json.dump([p.id for p in hm_samples], f)



if __name__ == '__main__':
    main()