import argparse
import lmdb
from pathlib import Path
import ujson as json

import visdom

from terial.classifier.utils import visualize_input

INPUT_SIZE = 224


vis = visdom.Visdom(env='test-dataset')

parser = argparse.ArgumentParser()
parser.add_argument(dest='snapshot_dir', type=Path)
args = parser.parse_args()


def main():
    train_path = Path(args.snapshot_dir, 'train')
    validation_path = Path(args.snapshot_dir, 'validation')
    meta_path = Path(args.snapshot_dir, 'meta.json')

    with meta_path.open('r') as f:
        meta_dict = json.load(f)

    env = lmdb.open(train_path, readonly=False)

    print(f"Woohoo")
    for batch in train_loader:
        vis.image(visualize_input(batch))


if __name__ == '__main__':
    main()

