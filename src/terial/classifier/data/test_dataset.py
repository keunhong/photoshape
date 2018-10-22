import argparse
from pathlib import Path
import ujson as json

import visdom
from torch.utils.data import DataLoader

from terial.classifier import transforms, rendering_dataset
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

    print(f"Opening LMDB datasets.")
    train_dataset = rendering_dataset.MaterialRendDataset(
        train_path,
        meta_dict,
        shape=(500, 500),
        image_transform=transforms.train_image_transform(INPUT_SIZE),
        mask_transform=transforms.train_mask_transform(INPUT_SIZE))

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        pin_memory=False,
        collate_fn=rendering_dataset.collate_fn)

    print(f"Woohoo")
    for batch in train_loader:
        vis.image(visualize_input(batch))


if __name__ == '__main__':
    main()

