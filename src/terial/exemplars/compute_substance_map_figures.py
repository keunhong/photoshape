import logging

from tqdm import tqdm

from kitnn.models import minc
from terial import config
from terial.database import session_scope
from terial.models import Exemplar
import toolbox.images


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

IMAGE_PAD_SIZE = 100
IMAGE_SHAPE = (500, 500)


def main():
    with session_scope(commit=False) as sess:
        exemplars = sess.query(Exemplar).order_by('id asc').all()


    # exemplars = [e for e in exemplars if
    #              not e.data_exists(config.EXEMPLAR_SUBST_MAP_NAME)]

    pbar = tqdm(exemplars)
    for exemplar in pbar:
        pbar.set_description(f'{exemplar.id}: loading')

        if not exemplar.data_exists(config.EXEMPLAR_SUBST_MAP_NAME):
            continue
        if exemplar.data_exists(config.EXEMPLAR_SUBST_VIS_NAME):
            continue

        try:
            subst_map = exemplar.load_data(config.EXEMPLAR_SUBST_MAP_NAME)
        except:
            logger.exception('Could not load')
            continue

        # image = exemplar.load_cropped_image()
        # image = toolbox.images.pad(image, IMAGE_PAD_SIZE, mode='edge')
        # fg_mask = toolbox.images.bright_pixel_mask(image, percentile=95)

        pbar.set_description(f'{exemplar.id}: computing substance map')
        subst_map_vis = toolbox.images.visualize_map(
            subst_map,
            bg_value=minc.REMAPPED_SUBSTANCES.index('background'),
            values=list(range(0, len(minc.REMAPPED_SUBSTANCES))))

        pbar.set_description(f'{exemplar.id}: saving data')

        exemplar.save_data(config.EXEMPLAR_SUBST_VIS_NAME, subst_map_vis)


if __name__ == '__main__':
    main()
