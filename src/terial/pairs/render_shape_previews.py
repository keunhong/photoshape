"""
Computes aligning features for all ShapeNet shapes.
"""
import logging
import numpy as np
import skimage
import visdom
import warnings
from tqdm import tqdm

from rendkit.graphics_utils import compute_tight_clipping_planes
from rendkit.shortcuts import render_preview, render_segments
from terial import config, controllers
from terial.models import ExemplarShapePair
from terial.database import session_scope
from toolbox import cameras
from toolbox.images import mask_bbox, crop_tight_fg

from vispy import app

app.use_app('glfw')

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logging.getLogger('rendkit').setLevel(logging.WARNING)


vis = visdom.Visdom(env='pair-generate-rends')


def main():
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs(
            sess,
            by_shape=True,
            by_shape_topk=5,
            order_by=ExemplarShapePair.distance.asc(),
        )

        print(f"Fetched {count} pairs")

        pbar = tqdm(pairs)
        for pair in pbar:
            # if pair.data_exists(config.SHAPE_REND_PREVIEW_NAME):
            #     continue
            render_model_exemplars(pbar, pair)


def render_model_exemplars(pbar, pair: ExemplarShapePair,
                           render_shape=config.SHAPE_REND_SHAPE):
    camera = cameras.spherical_coord_to_cam(
        pair.fov, pair.azimuth, pair.elevation)

    pbar.set_description(f'[{pair.id}] Loading shape')
    mesh, materials = pair.shape.load()
    camera.near, camera.far = compute_tight_clipping_planes(mesh, camera.view_mat())

    segment_im = render_segments(mesh, camera)
    fg_bbox = mask_bbox(segment_im > -1)

    pbar.set_description('Rendering preview')
    phong_im = np.clip(
        render_preview(mesh, camera,
                       config.SHAPE_REND_RADMAP_PATH,
                       gamma=2.2, ssaa=2), 0, 1)
    phong_im = crop_tight_fg(phong_im, render_shape, bbox=fg_bbox)

    vis.image(phong_im.transpose((2, 0, 1)), win='shape-preview')

    pbar.set_description(f'[{pair.id}] Saving data')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pair.save_data(config.SHAPE_REND_PREVIEW_NAME,
                       skimage.img_as_uint(phong_im))


if __name__ == '__main__':
    main()
