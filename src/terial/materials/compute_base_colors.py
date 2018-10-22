import numpy as np
from skimage.color import rgb2lab
from skimage.io import imread
from tqdm import tqdm

from terial.database import session_scope
from terial import models
from pathlib import Path
from toolbox.images import linear_to_srgb
from toolbox.colors import (compute_lab_histogram, lab_rgb_gamut_bin_mask,
                            compute_rgb_histogram)
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--space', choices=['rgb', 'lab'], required=True)
parser.add_argument('--shape', default='3,10,10', required=True)
parser.add_argument('--sigma', default='0.5,1,1', required=True)
args = parser.parse_args()


num_bins = tuple(map(int, args.shape.split(',')))
sigma = tuple(map(float, args.sigma.split(',')))
key = f'base_color_hist_{args.space}_{num_bins[0]}_{num_bins[1]}_{num_bins[2]}'


def main():
    with session_scope() as sess:
        materials = sess.query(models.Material).all()

        for material in tqdm(materials):
            base_color = material.load_base_color()
            image = base_color.clip(0, 1)

            if args.space == 'lab':
                valid_bin_mask, _ = lab_rgb_gamut_bin_mask(num_bins)
                hist = compute_lab_histogram(image,
                                             num_bins=num_bins,
                                             sigma=sigma)
            elif args.space == 'rgb':
                hist = compute_rgb_histogram(image,
                                             num_bins=num_bins,
                                             sigma=sigma)
            else:
                raise ValueError(f'Unknown space {args.space}')

            params = material.params if material.params else {}

            material.params = {
                **params,
                key: {
                    'space': args.space,
                    'shape': num_bins,
                    'sigma': sigma,
                    'hist': hist.tolist(),
                },
            }
        sess.commit()


if __name__ == '__main__':
    main()