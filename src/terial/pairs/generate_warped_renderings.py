"""
Computes aligning features for all ShapeNet shapes.
"""
import argparse
import logging
import os
import warnings

import visdom
from pydensecrf import densecrf
import numpy as np

import click as click
from skimage import transform
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.morphology import binary_closing, disk
from tqdm import tqdm

from kitnn.colors import tensor_to_image, rgb_to_lab, image_to_tensor
from kitnn.utils import softmax2d
from terial import alignment, config, controllers, visutils
from terial.config import SUBSTANCES
from terial.flow import apply_flow
from terial.models import ExemplarShapePair
from terial.database import session_scope
from terial.pairs import utils
from toolbox.images import visualize_map, resize

from vispy import app
app.use_app('glfw')


vis = visdom.Visdom(env='generate-warped-rends')

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
logging.getLogger('rendkit').setLevel(logging.WARNING)


parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
args = parser.parse_args()


def parse_rend_filename(fname):
    fname, _ = os.path.splitext(fname)
    s = [s.split('=') for s in fname.split(',')]
    return {k: v for k, v in s}


def compute_features(path):
    image = imread(path)
    return alignment.compute_features(
        image, bin_size=config.ALIGN_BIN_SIZE, im_shape=config.ALIGN_IM_SHAPE)


def main():
    warnings.simplefilter("ignore")

    filters = []
    if args.category:
        filters.append(ExemplarShapePair.shape.has(category=args.category))

    print(f"Fetching pairs")
    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs_default(sess, filters=filters)
        # pairs, count = controllers.fetch_pairs(
        #     sess,
        #     by_shape=True,
        #     order_by=ExemplarShapePair.distance.asc(),
        # )

        print(f"Fetched {len(pairs)} pairs.")

        pairs = [
            pair for pair in pairs
        ]

        print(f"Computing warped renderings for {len(pairs)} pairs.")

        pbar = tqdm(pairs)
        for pair in pbar:
            pbar.set_description(f'Pair {pair.id}, Exemplar {pair.exemplar.id}')
            if not pair.data_exists(config.SHAPE_REND_SEGMENT_MAP_NAME):
                tqdm.write('No segment map')
                continue
            if not pair.data_exists(config.FLOW_DATA_NAME):
                tqdm.write('No flow')
                continue
            if pair.data_exists(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME):
                # tqdm.write('Exists')
                continue
            warp_renderings(pbar, pair)


def warp_renderings(pbar, pair: ExemplarShapePair):
    if not pair.data_exists(config.FLOW_DATA_NAME):
        logger.error('pair %d does not have flow', pair.id)
        return
    flow = pair.load_data(config.FLOW_DATA_NAME)
    vx, vy = flow[:, :, 0], flow[:, :, 1]
    # phong_im = pair.load_data(config.SHAPE_REND_PHONG_NAME)
    # pair.save_data(config.PAIR_SHAPE_WARPED_PHONG_NAME,
    #                apply_flow(phong_im, vx, vy))

    shape_seg_map = pair.load_data(config.SHAPE_REND_SEGMENT_MAP_NAME) - 1
    warped_seg_map = apply_flow(shape_seg_map, vx, vy)

    image = transform.resize(pair.exemplar.load_cropped_image(),
                             config.SHAPE_REND_SHAPE, anti_aliasing=True,
                             mode='reflect')
    crf_seg_map = apply_segment_crf(image, warped_seg_map)

    subst_map = pair.exemplar.load_data(config.EXEMPLAR_SUBST_MAP_NAME)
    subst_map = resize(subst_map, crf_seg_map.shape[:2], order=0)

    seg_subst_ids = utils.compute_substance_ids_by_segment(
        subst_map, crf_seg_map)
    proxy_subst_map = utils.compute_segment_substance_map(
        crf_seg_map, seg_subst_ids)
    shape_subst_map = utils.compute_segment_substance_map(
        shape_seg_map, seg_subst_ids)

    shape_seg_vis = visutils.visualize_segment_map(shape_seg_map)
    warped_seg_vis = visutils.visualize_segment_map(warped_seg_map)
    crf_seg_vis = visutils.visualize_segment_map(crf_seg_map)

    exemplar_subst_vis = visutils.visualize_substance_map(subst_map)
    proxy_subst_vis = visutils.visualize_substance_map(proxy_subst_map)
    shape_subst_vis = visutils.visualize_substance_map(shape_subst_map)

    vis.image(pair.exemplar.load_cropped_image().transpose((2, 0, 1)),
              win='exemplar-image')
    vis.image(crf_seg_vis.transpose((2, 0, 1)), win='segment-map',
              opts={'title': 'segment-map'})
    vis.image(exemplar_subst_vis.transpose((2, 0, 1)), win='subst-map',
              opts={'title': 'subst-map'})
    vis.image(proxy_subst_vis.transpose((2, 0, 1)), win='proxy-subst-map',
              opts={'title': 'proxy-subst-map'})
    vis.image(shape_subst_vis.transpose((2, 0, 1)), win='shape-subst-map',
              opts={'title': 'shape-subst-map'})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pair.save_data(config.PAIR_PROXY_SUBST_MAP_NAME, proxy_subst_map)
        pair.save_data(config.PAIR_PROXY_SUBST_VIS_NAME, proxy_subst_vis)

        pair.save_data(config.PAIR_SHAPE_SUBST_MAP_NAME, shape_subst_map)
        pair.save_data(config.PAIR_SHAPE_SUBST_VIS_NAME, shape_subst_vis)

        pair.save_data(config.PAIR_SHAPE_WARPED_SEGMENT_MAP_NAME,
                       (warped_seg_map + 1).astype(np.uint8))
        pair.save_data(config.PAIR_SHAPE_WARPED_SEGMENT_VIS_NAME,
                       warped_seg_vis)
        pair.save_data(config.SHAPE_REND_SEGMENT_VIS_NAME, shape_seg_vis)

        pair.save_data(config.PAIR_SHAPE_CLEAN_SEGMENT_VIS_NAME, crf_seg_vis)
        pair.save_data(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME,
                       (crf_seg_map + 1).astype(np.uint8))

    ov_fig_warped = (rgb2gray(image)[:, :, None].repeat(3, 2)
                     + visualize_map(warped_seg_map)) / 2.0
    ov_fig_original = (rgb2gray(image)[:, :, None].repeat(3, 2)
                       + visualize_map(shape_seg_map)) / 2.0
    ov_fig_clean = (rgb2gray(image)[:, :, None].repeat(3, 2)
                    + visualize_map(crf_seg_map)) / 2.0
    ov_fig = np.hstack(
        (ov_fig_original, ov_fig_warped, ov_fig_clean))

    vis.image(ov_fig.transpose((2, 0, 1)), win='overlay')

    pair.save_data(config.PAIR_SEGMENT_OVERLAY_NAME, ov_fig)
    tqdm.write(f' * Saved for pair {pair.id}')


def apply_segment_crf(image, segment_map, theta_p=0.05, theta_L=5, theta_ab=5):
    image_lab = tensor_to_image(rgb_to_lab(image_to_tensor(image)))
    perc = np.percentile(np.unique(image[:, :, :3].min(axis=2)), 98)
    bg_mask = np.all(image > perc, axis=2)
    vis.image(bg_mask.astype(np.uint8) * 255, win='bg-mask')

    p_y, p_x = np.mgrid[0:image_lab.shape[0], 0:image_lab.shape[1]]

    feats = np.zeros((5, *image_lab.shape[:2]), dtype=np.float32)
    d = min(image_lab.shape[:2])
    feats[0] = p_x / (theta_p * d)
    feats[1] = p_y / (theta_p * d)
    # feats[2] = fg_mask / 50
    feats[2] = image_lab[:, :, 0] / theta_L
    feats[3] = image_lab[:, :, 1] / theta_ab
    feats[4] = image_lab[:, :, 2] / theta_ab
    # vals = [v for v in np.unique(segment_map) if v >= 0]
    vals = np.unique(segment_map)
    probs = np.zeros((*segment_map.shape, len(vals)))
    for i, val in enumerate(vals):
        probs[:, :, i] = segment_map == val
    probs[bg_mask, 0] = 3
    probs[~bg_mask & (segment_map == -1)] = 1 / (len(vals))
    probs = softmax2d(probs)

    # for c in range(probs.shape[2]):
    #     vis.image(probs[:, :, c], win=f'prob-{c}', opts={'title': f'prob-{c}'})

    crf = densecrf.DenseCRF2D(*probs.shape)
    unary = np.rollaxis(
        -np.log(probs), axis=-1).astype(dtype=np.float32, order='c')
    crf.setUnaryEnergy(np.reshape(unary, (probs.shape[-1], -1)))
    crf.addPairwiseEnergy(np.reshape(feats, (feats.shape[0], -1)),
                          compat=3)

    Q = crf.inference(20)
    Q = np.array(Q).reshape((-1, *probs.shape[:2]))
    probs = np.rollaxis(Q, 0, 3)

    cleaned_seg_ind_map = probs.argmax(axis=-1)
    cleaned_seg_map = np.full(cleaned_seg_ind_map.shape,
                              fill_value=-1, dtype=int)
    for ind in np.unique(cleaned_seg_ind_map):
        cleaned_seg_map[cleaned_seg_ind_map == ind] = vals[ind]
    return cleaned_seg_map


if __name__ == '__main__':
    main()