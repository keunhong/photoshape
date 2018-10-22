import argparse
import logging

import warnings

import visdom
from tqdm import tqdm
import sqlalchemy as sa

import numpy as np
from kitnn.models import minc
from terial import config
from terial.config import MINC_VGG16_WEIGHTS_PATH
from terial.database import session_scope
from terial.models import Exemplar
import toolbox.images


vis = visdom.Visdom(env='compute-substance-maps')


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

IMAGE_PAD_SIZE = 25
IMAGE_SHAPE = (500, 500)

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default=None)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=99999999999)
args = parser.parse_args()


def main():
    filters = [
        Exemplar.id >= args.start,
        Exemplar.id < args.end,
    ]

    if args.category:
        filters.append(Exemplar.category == args.category)

    with session_scope(commit=False) as sess:
        exemplars = sess.query(Exemplar) \
            .filter(sa.and_(*filters)) \
            .order_by(Exemplar.id.asc()).all()

    logger.info('Loading VGG.')
    mincnet = minc.MincVGG()
    mincnet.load_npy(MINC_VGG16_WEIGHTS_PATH)
    mincnet = mincnet.cuda()

    # exemplars = [e for e in exemplars if
    #              not e.data_exists(config.EXEMPLAR_SUBST_MAP_NAME)]

    pbar = tqdm(exemplars)
    for exemplar in pbar:
        pbar.set_description(f'{exemplar.id}: loading')

        if exemplar.data_exists(config.EXEMPLAR_SUBST_MAP_NAME):
            try:
                exemplar.load_data(config.EXEMPLAR_SUBST_MAP_NAME)
                continue
            except:
                logger.exception('Could not load')

        image = exemplar.load_cropped_image()
        image = toolbox.images.pad(image, IMAGE_PAD_SIZE, mode='constant')
        fg_mask = toolbox.images.bright_pixel_mask(image, percentile=99)

        subst_map, _ = compute_substance_map(pbar, mincnet, image, fg_mask)
        # subst_map = toolbox.images.apply_mask(
        #     subst_map, fg_mask, fill=minc.REMAPPED_SUBSTANCES.index('background'))
        subst_map = toolbox.images.unpad(subst_map, IMAGE_PAD_SIZE)
        subst_map_vis = toolbox.images.visualize_map(
            subst_map,
            bg_value=minc.REMAPPED_SUBSTANCES.index('background'),
            values=list(range(0, len(minc.REMAPPED_SUBSTANCES))))

        pbar.set_description(f'{exemplar.id}: saving data')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exemplar.save_data(config.EXEMPLAR_SUBST_MAP_NAME,
                               subst_map.astype(np.uint8))


def compute_substance_map(pbar, mincnet, image, fg_mask):
    processed_image = minc.preprocess_image(image)

    s = 1
    prob_maps, feat_dicts = minc.compute_probs_multiscale(
        processed_image, mincnet, use_cuda=True,
        scales=[1.0*s, 1.414*s, 1/1.414*s])
    prob_map_avg = minc.combine_probs(prob_maps, processed_image,
                                      remap=True, fg_mask=fg_mask)

    pbar.set_description("Running dense CRF")
    prob_map_crf = minc.compute_probs_crf(image, prob_map_avg)
    pbar.set_description("Resizing {} => {}"
          .format(prob_map_crf.shape[:2], processed_image.shape[:2]))
    prob_map_crf = toolbox.images.resize(
        prob_map_crf, processed_image.shape[:2], order=2)
    subst_id_map = np.argmax(prob_map_crf, axis=-1)
    return subst_id_map, prob_map_crf


if __name__ == '__main__':
    main()
