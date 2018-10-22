"""
Computes shape to exemplar flows.
"""
import os
import logging
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import numpy as np
import click as click
import matlab.engine
from skimage import transform
from skimage.color import rgb2gray
from skimage.io import imsave
from skimage.morphology import disk, binary_closing

from tqdm import tqdm
from visdom import Visdom

from terial import config, controllers
from terial.flow import resize_flow, visualize_flow, apply_flow
from terial.models import ExemplarShapePair
from terial.database import session_scope

vis = Visdom(env='compute-flows')


def bright_pixel_mask(image, percentile=80):
    image = rgb2gray(image)
    perc = np.percentile(np.unique(image), percentile)
    mask = image < perc
    return mask


@click.command()
def main():
    warnings.simplefilter("error", UserWarning)

    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs(
            sess,
            max_dist=config.ALIGN_DIST_THRES,
            order_by=ExemplarShapePair.distance.asc(),
        )

        print(f"Fetched {count} pairs")

        base_pattern = np.dstack((
            np.zeros(config.SHAPE_REND_SHAPE), *np.meshgrid(
                np.linspace(0, 1, config.SHAPE_REND_SHAPE[0]),
                np.linspace(0, 1, config.SHAPE_REND_SHAPE[1]))))


        pbar = tqdm(pairs)
        pair: ExemplarShapePair
        for pair in pbar:
            pbar.set_description(f'Pair {pair.id}')
            # if not pair.exemplar.data_exists(config.EXEMPLAR_SUBST_MAP_NAME, type='numpy'):
            #     logger.warning('pair %d does not have substance map', pair.id)
            #     continue

            if not pair.data_exists(config.FLOW_DATA_NAME):
                print(f'Pair {pair.id} does not have flow')
                continue

            exemplar_sil = bright_pixel_mask(
                pair.exemplar.load_cropped_image(), percentile=95)
            exemplar_sil = binary_closing(exemplar_sil, selem=disk(3))
            exemplar_sil = transform.resize(exemplar_sil, (500, 500),
                                            anti_aliasing=True, mode='reflect')
            shape_sil = pair.load_data(config.SHAPE_REND_SEGMENT_MAP_NAME) - 1
            shape_sil = (shape_sil > -1)
            shape_sil = binary_closing(shape_sil, selem=disk(3))

            exemplar_sil_im = exemplar_sil[:, :, None].repeat(repeats=3, axis=2).astype(float)
            shape_sil_im = shape_sil[:, :, None].repeat(repeats=3, axis=2).astype(float)

            exemplar_sil_im[exemplar_sil == 0] = base_pattern[exemplar_sil == 0]
            shape_sil_im[shape_sil == 0] = base_pattern[shape_sil == 0]

            pair.save_data(config.FLOW_SHAPE_SILHOUETTE_VIS, shape_sil_im)
            pair.save_data(config.FLOW_EXEMPLAR_SILHOUETTE_VIS, exemplar_sil_im)


if __name__ == '__main__':
    main()