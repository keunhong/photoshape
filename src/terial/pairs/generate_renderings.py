"""
Computes aligning features for all ShapeNet shapes.
"""
import argparse
import logging
import os

import numpy as np
import skimage
import visdom
import warnings
from skimage.io import imread
from tqdm import tqdm

import toolbox.images
from rendkit.graphics_utils import compute_tight_clipping_planes
from rendkit.shortcuts import (render_segments, render_wavefront_mtl,
                               render_mesh_normals)
from terial import alignment, config, controllers
from terial.models import ExemplarShapePair
from terial.database import session_scope
from toolbox import cameras
from toolbox.images import mask_bbox, crop_tight_fg, visualize_map

from vispy import app

app.use_app('glfw')

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logging.getLogger('rendkit').setLevel(logging.WARNING)


parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
args = parser.parse_args()


vis = visdom.Visdom(env='pair-generate-rends')


def parse_rend_filename(fname):
    fname, _ = os.path.splitext(fname)
    s = [s.split('=') for s in fname.split(',')]
    return {k: v for k, v in s}


def compute_features(path):
    image = imread(path)
    return alignment.compute_features(
        image, bin_size=config.ALIGN_BIN_SIZE, im_shape=config.ALIGN_IM_SHAPE)


def main():
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    filters = []
    if args.category:
        filters.append(ExemplarShapePair.shape.has(category=args.category))

    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs_default(sess, filters=filters)
        # shapes, _ = controllers.fetch_shapes_with_pairs(sess)
        # print(f"Loading shapes")
        # pairs = []
        # for shape in tqdm(shapes):
        #     pairs.extend(shape.get_topk_pairs(config.INFERENCE_TOPK,
        #                                       config.INFERENCE_MAX_DIST))

        # pairs, count = controllers.fetch_pairs(
        #     sess,
        #     by_shape=True,
        #     order_by=ExemplarShapePair.shape_id.asc(),
        # )

        print(f"Fetched {len(pairs)} pairs")

        pairs = [
            pair for pair in pairs
            # if not pair.data_exists(config.SHAPE_REND_SEGMENT_MAP_NAME)
            if not pair.data_exists(config.PAIR_FG_BBOX_NAME)
        ]

        print(f"Generating renderings for {len(pairs)} pairs.")

        pbar = tqdm(pairs)
        for pair in pbar:
            # if pair.data_exists(config.SHAPE_REND_SEGMENT_MAP_NAME):
            if pair.data_exists(config.PAIR_FG_BBOX_NAME):
                continue
            render_model_exemplars(pbar, pair)


def render_model_exemplars(pbar, pair: ExemplarShapePair,
                           render_shape=config.SHAPE_REND_SHAPE):
    warnings.simplefilter('ignore')

    camera = cameras.spherical_coord_to_cam(
        pair.fov, pair.azimuth, pair.elevation)

    pbar.set_description(f'[{pair.id}] Loading shape')
    mesh, materials = pair.shape.load()
    camera.near, camera.far = compute_tight_clipping_planes(mesh, camera.view_mat())

    pbar.set_description(f'[{pair.id}] Rendering segments')
    if not pair.data_exists(config.PAIR_FG_BBOX_NAME):
        segment_im = render_segments(mesh, camera)
        fg_mask = segment_im > -1
        fg_bbox = mask_bbox(fg_mask)
        segment_im = crop_tight_fg(
            segment_im, render_shape,
            bbox=fg_bbox, fill=-1, order=0)
        pair.save_data(config.SHAPE_REND_SEGMENT_VIS_NAME,
                       skimage.img_as_uint(visualize_map(segment_im)))
        pair.save_data(config.SHAPE_REND_SEGMENT_MAP_NAME,
                       (segment_im + 1).astype(np.uint8))

        tqdm.write(f" * Saving {pair.get_data_path(config.PAIR_FG_BBOX_NAME)}")
        pair.save_data(config.PAIR_RAW_SEGMENT_MAP_NAME,
                       (segment_im + 1).astype(np.uint8))
        pair.save_data(config.PAIR_FG_BBOX_NAME, fg_mask.astype(np.uint8) * 255)
    else:
        fg_mask = pair.load_data(config.PAIR_FG_BBOX_NAME)
        fg_bbox = mask_bbox(fg_mask)

    if not pair.data_exists(config.SHAPE_REND_PHONG_NAME):
        pbar.set_description('Rendering phong')
        phong_im = np.clip(
            render_wavefront_mtl(mesh, camera, materials,
                                 config.SHAPE_REND_RADMAP_PATH,
                                 gamma=2.2, ssaa=3, tonemap='reinhard'), 0, 1)
        phong_im = crop_tight_fg(phong_im, render_shape, bbox=fg_bbox)
        pbar.set_description(f'[{pair.id}] Saving data')
        pair.save_data(config.SHAPE_REND_PHONG_NAME,
                       skimage.img_as_uint(phong_im))

    # pbar.set_description('Rendering normals')
    # normal_map = crop_tight_fg(
    #     render_mesh_normals(mesh, camera), render_shape,
    #     bbox=fg_bbox, fill=0)
    # # normal_map = normal_map.clip(-1, 1)
    # normal_map = toolbox.images.to_8bit((normal_map + 1) / 2)
    # pair.save_data(config.SHAPE_REND_NORMALS_NAME, normal_map)

    # pbar.set_description('Rendering UVs')
    # uvs_im = crop_tight_fg(
    #     render_uvs(mesh, camera), render_shape,
    #     bbox=fg_bbox, fill=0)

    # pbar.set_description(f'[{pair.id}] Rendering Tangents')
    # tangents_im = crop_tight_fg(
    #     render_tangents(mesh, camera), render_shape,
    #     bbox=fg_bbox, fill=0)
    # bitangents_im = crop_tight_fg(
    #     render_bitangents(mesh, camera), render_shape,
    #     bbox=fg_bbox, fill=0)
    #
    # pbar.set_description(f'[{pair.id}] Rendering World Coords')
    # world_coords_im = crop_tight_fg(
    #     render_world_coords(mesh, camera), render_shape,
    #     bbox=fg_bbox, fill=0)
    #
    # pbar.set_description(f'[{pair.id}] Rendering Depth')
    # depth_im = crop_tight_fg(
    #     render_depth(mesh, camera), render_shape,
    #     bbox=fg_bbox, fill=0)


    # pair.save_data(config.SHAPE_REND_NORMALS_NAME,
    #                skimage.img_as_uint(abs(normals_im).clip(0, 1)), type='image')
    # pair.save_data(config.SHAPE_REND_NORMALS_NAME, normals_im, type='numpy')

    # pair.save_data(config.SHAPE_REND_UV_U_NAME,
    #                skimage.img_as_uint(uvs_im[:, :, 0]), type='image')
    # pair.save_data(config.SHAPE_REND_UV_V_NAME,
    #                skimage.img_as_uint(uvs_im[:, :, 1]), type='image')

    # pair.save_data(config.SHAPE_REND_UV_U_NAME, uvs_im[:, :, 0], type='numpy')
    # pair.save_data(config.SHAPE_REND_UV_V_NAME, uvs_im[:, :, 1], type='numpy')
    #
    # pair.save_data(config.SHAPE_TANGENTS_NAME,
    #                abs(tangents_im).clip(0, 1), type='image')
    # pair.save_data(config.SHAPE_TANGENTS_NAME, tangents_im, type='numpy')
    # pair.save_data(config.SHAPE_BITANGENTS_NAME,
    #                abs(bitangents_im).clip(0, 1), type='image')
    # pair.save_data(config.SHAPE_BITANGENTS_NAME, bitangents_im, type='numpy')
    #
    # world_coords_vis = ((world_coords_im - world_coords_im.min())
    #                     / (world_coords_im.max() - world_coords_im.min()))
    # pair.save_data(config.SHAPE_WORLD_COORDS_NAME,
    #                world_coords_vis.clip(0, 1), type='image')
    # pair.save_data(config.SHAPE_WORLD_COORDS_NAME, world_coords_im, type='numpy')
    #
    # pair.save_data(config.SHAPE_DEPTH_NAME,
    #                depth_im.clip(0, 1), type='image')
    # pair.save_data(config.SHAPE_DEPTH_NAME, depth_im, type='numpy')


if __name__ == '__main__':
    main()
