import argparse
import json

from pathlib import Path

import logging

import visdom
from sqlalchemy import orm

import brender
import toolbox.io
import toolbox.io.images
import vispy.app
from terial import config, models
from terial.classifier.rendering import blender
from terial.classifier.inference.utils import compute_weighted_scores
from terial.database import session_scope
from terial.models import ExemplarShapePair
import toolbox.images

vispy.app.use_app('glfw')
vis = visdom.Visdom(env='brdf-classifier-render-inferred')

logger = logging.getLogger(__name__)

_REND_SHAPE = (1280, 1280)
_FINAL_SHAPE = (500, 500)


parser = argparse.ArgumentParser()
parser.add_argument(dest='inference_path', type=Path)
parser.add_argument(dest='out_path', type=Path)
parser.add_argument('--type', default='inferred',
                    choices=['inferred', 'mtl'])
parser.add_argument('--use-weighted-scores', action='store_true')
args = parser.parse_args()


def main():
    if not args.inference_path.exists():
        print(f'{args.inference_path!s} does not exist.')
        return

    with args.inference_path.open('r') as f:
        inference_dict = json.load(f)

    with session_scope() as sess:
        pair = (sess.query(ExemplarShapePair)
                .options(orm.joinedload(models.ExemplarShapePair.exemplar),
                         orm.joinedload(models.ExemplarShapePair.shape))
                .get(inference_dict['pair_id']))
        envmap = sess.query(models.Envmap).get(13)
        materials = sess.query(models.Material).all()
        mat_by_id = {m.id: m for m in materials}

    print(f" * Loading {pair.get_data_path(config.PAIR_FG_BBOX_NAME)}")
    fg_mask = pair.load_data(config.PAIR_FG_BBOX_NAME)
    fg_mask = toolbox.images.resize(fg_mask, _REND_SHAPE)
    fg_mask = fg_mask.astype(bool)
    fg_bbox = toolbox.images.mask_bbox(fg_mask)
    crop_bbox = toolbox.images.bbox_make_square(fg_bbox)

    if args.use_weighted_scores:
        compute_weighted_scores(inference_dict, mat_by_id, sort=True)

    app = brender.Brender()
    app.init()
    scene = blender.construct_inference_scene(
        app, pair, inference_dict, mat_by_id, envmap,
        num_samples=128,
        tile_size=(640, 640),
        rend_shape=_REND_SHAPE)
    logger.info(' * Rendering...')
    rend = scene.render_to_array(format='exr')
    print('aaa', rend.shape, fg_mask.shape)
    rend_srgb = toolbox.images.linear_to_srgb(rend)
    rend_srgb = toolbox.images.crop_bbox(rend_srgb, crop_bbox)
    print(f" * Saving to {args.out_path!s}")
    toolbox.io.images.save_image(args.out_path, rend_srgb)


if __name__ == '__main__':
    main()
