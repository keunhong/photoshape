"""
Computes aligning features for all ShapeNet shapes.
"""
import logging
import math
import os

import click as click
import numpy as np
import skimage
import visdom
import warnings
from skimage.io import imread
from tqdm import tqdm

from rendkit.camera import PerspectiveCamera
from rendkit.graphics_utils import compute_tight_clipping_planes
from rendkit.shortcuts import (render_segments, render_wavefront_mtl,
                               render_material_dicts)
from terial import alignment, config, controllers
from terial.config import SUBSTANCES
from terial.models import ExemplarShapePair
from terial.database import session_scope
from terial.pairs.utils import compute_segment_substances
from toolbox import cameras
from toolbox.cameras import spherical_to_cartesian
from toolbox.images import mask_bbox, crop_tight_fg, visualize_map, QUAL_COLORS

from vispy import app

app.use_app('glfw')

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logging.getLogger('rendkit').setLevel(logging.WARNING)


vis = visdom.Visdom(env='pair-subst-rends')


def parse_rend_filename(fname):
    fname, _ = os.path.splitext(fname)
    s = [s.split('=') for s in fname.split(',')]
    return {k: v for k, v in s}


def compute_features(path):
    image = imread(path)
    return alignment.compute_features(
        image, bin_size=config.ALIGN_BIN_SIZE, im_shape=config.ALIGN_IM_SHAPE)


@click.command()
def main():
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs(
            sess,
            by_shape=False,
            order_by=ExemplarShapePair.distance.asc(),
        )

        print(f"Fetched {count} pairs")

        pbar = tqdm(pairs)
        for pair in pbar:
            render_model_exemplars(pbar, pair)


def render_model_exemplars(pbar, pair: ExemplarShapePair,
                           render_shape=config.SHAPE_REND_SHAPE):
    pbar.set_description(f'[{pair.id}] Loading shape')
    mesh, materials = pair.shape.load()

    dist = 150
    camera = PerspectiveCamera(
        size=(1000, 1000), fov=0, near=0.1, far=5000.0,
        position=(0, 0, -dist), clear_color=(1, 1, 1, 0),
        lookat=(0, 0, 0), up=(0, 1, 0))
    camera.position = spherical_to_cartesian(dist, 1.2 + math.pi, 1.33)
    camera.fov = 50

    try:
        seg_substances = compute_segment_substances(pair)
    except FileNotFoundError:
        return

    material_dict = {}
    for seg_name, substance in seg_substances.items():
        subst_idx = SUBSTANCES.index(substance)
        material_dict[seg_name]  = {
            'type': 'blinn_phong',
            'diffuse': tuple(c/255
                             for c in QUAL_COLORS[subst_idx % len(QUAL_COLORS)]),
            'specular': (0.1, 0.1, 0.1),
            'roughness': 0.1,
        }

    pbar.set_description(f'[{pair.id}] Rendering segments')
    rend_im = np.clip(
        render_material_dicts(mesh, camera, material_dict,
                             config.RADMAP_DIR / 'k_studio_dimmer.cross.exr',
                             gamma=2.2, ssaa=2, format='rgba'), 0, 1)

    vis.image(rend_im.transpose((2, 0, 1)), win='preview')

    pbar.set_description(f'[{pair.id}] Saving data')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pair.save_data(config.SHAPE_REND_SUBSTANCE_NAME,
                       skimage.img_as_uint(rend_im))



if __name__ == '__main__':
    main()
