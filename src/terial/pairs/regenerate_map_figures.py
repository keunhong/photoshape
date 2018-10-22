import logging

from tqdm import tqdm

from kitnn.models import minc
from terial import config, controllers
from terial.database import session_scope
from terial.models import Exemplar, ExemplarShapePair
import toolbox.images


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

IMAGE_PAD_SIZE = 100
IMAGE_SHAPE = (500, 500)


def main():
    with session_scope(commit=False) as sess:
        pairs, count = controllers.fetch_pairs(
            sess,
            max_dist=config.ALIGN_DIST_THRES,
            order_by=ExemplarShapePair.distance.asc())


    pbar = tqdm(pairs)
    for pair in pbar:
        pbar.set_description(f'{pair.id}: loading')

        if pair.data_exists(config.SHAPE_REND_SEGMENT_MAP_NAME):
            seg_map = pair.load_data(config.SHAPE_REND_SEGMENT_MAP_NAME) - 1
            seg_map_vis = toolbox.images.visualize_map(
                seg_map,
                bg_value=-1,
                values=range(-1, seg_map.max() + 1))

            pbar.set_description(f'{pair.id}: saving seg map vis')

            pair.save_data(config.SHAPE_REND_SEGMENT_VIS_NAME, seg_map_vis)

        if pair.data_exists(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME):
            seg_map = pair.load_data(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME) - 1
            seg_map_vis = toolbox.images.visualize_map(
                seg_map,
                bg_value=-1,
                values=range(-1, seg_map.max() + 1))

            pbar.set_description(f'{pair.id}: saving clean seg map vis')

            pair.save_data(config.PAIR_SHAPE_CLEAN_SEGMENT_VIS_NAME, seg_map_vis)

        if pair.data_exists(config.PAIR_SHAPE_WARPED_SEGMENT_MAP_NAME):
            seg_map = pair.load_data(config.PAIR_SHAPE_WARPED_SEGMENT_MAP_NAME) - 1
            seg_map_vis = toolbox.images.visualize_map(
                seg_map,
                bg_value=-1,
                values=range(-1, seg_map.max() + 1))

            pbar.set_description(f'{pair.id}: saving warped seg map vis')

            pair.save_data(config.PAIR_SHAPE_WARPED_SEGMENT_VIS_NAME, seg_map_vis)

if __name__ == '__main__':
    main()
