import json

from pathlib import Path

import bpy

import brender
from terial import models
from terial.classifier.rendering import blender
from terial.classifier.inference.utils import compute_weighted_scores
from terial.database import session_scope
from terial.models import ExemplarShapePair


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest='inference_path', type=Path)
parser.add_argument(dest='out_path', type=Path)
parser.add_argument('--pack-assets', action='store_true')
parser.add_argument('--type', default='inferred',
                    choices=['inferred', 'mtl'])
parser.add_argument('--animate', action='store_true')
parser.add_argument('--use-weighted-scores', action='store_true')
parser.add_argument('--use-minc-substances', action='store_true')
parser.add_argument('--frontal', action='store_true')
parser.add_argument('--diagonal', action='store_true')
parser.add_argument('--no-floor', action='store_true')
args = parser.parse_args()


if args.animate:
    _REND_SHAPE = (1080, 1920)
else:
    _REND_SHAPE = (1280, 1280)


def main():
    app = brender.Brender()

    if not args.inference_path.exists():
        print(f' * {args.inference_path!s} does not exist.')
        return

    with args.inference_path.open('r') as f:
        inference_dict = json.load(f)

    with session_scope() as sess:
        envmap = sess.query(models.Envmap).get(30)
        materials = sess.query(models.Material).all()
        mat_by_id = {m.id: m for m in materials}

        pair = sess.query(ExemplarShapePair).get(inference_dict['pair_id'])

        # if args.use_weighted_scores:
        #     compute_weighted_scores(inference_dict, mat_by_id,
        #                             force_substances=args.use_minc_substances,
        #                             sort=True)

        if args.use_weighted_scores:
            compute_weighted_scores(inference_dict, mat_by_id, sort=True,
                                    force_substances=True)

        app.init()
        scene = blender.construct_inference_scene(
            app, pair, inference_dict, mat_by_id, envmap, scene_type=args.type,
            rend_shape=_REND_SHAPE,
            frontal_camera=args.frontal,
            diagonal_camera=args.diagonal,
            add_floor=not args.no_floor)
        if args.animate:
            blender.animate_scene(scene)

        print(f' * Saving blend file to {args.out_path!s}')
        if args.pack_assets:
            bpy.ops.file.pack_all()
        else:
            bpy.ops.file.make_paths_absolute()
        bpy.ops.wm.save_as_mainfile(filepath=str(args.out_path))

        scene.clear_bmats()


if __name__ == '__main__':
    main()
