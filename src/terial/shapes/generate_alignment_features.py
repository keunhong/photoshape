"""
Computes aligning features for all shapes.
"""
import argparse
import logging
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sqlalchemy as sa

import numpy as np
from skimage.io import imread
from tqdm import tqdm

from terial import alignment, config
from terial.models import Shape
from terial.database import session_scope


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str)
args = parser.parse_args()


EXEMPLAR_SIZE = (100, 100)


def parse_rend_filename(fname):
    fname, _ = os.path.splitext(fname)
    s = [s.split('=') for s in fname.split(',')]
    return {k: v for k, v in s}


def compute_features(path):
    image = imread(path)
    return alignment.compute_features(
        image, bin_size=config.ALIGN_BIN_SIZE, im_shape=config.ALIGN_IM_SHAPE)


def main():
    filters = []
    if args.category:
        filters.append(Shape.category == args.category)


    with session_scope() as sess:
        shapes = (sess.query(Shape)
                  .filter(sa.and_(*filters))
                  .order_by(Shape.id.asc())
                  .all())

    pbar = tqdm(shapes)
    for shape in pbar:
        rend_dir = Path(shape.get_image_dir_path(), 'alignment/renderings')
        if not rend_dir.exists():
            continue
        if shape.data_exists(config.SHAPE_ALIGN_DATA_NAME):
            logger.warning(
                f'{config.SHAPE_ALIGN_DATA_NAME} exists for shape {shape.id}')
            continue

        rend_paths = sorted(rend_dir.iterdir())

        pbar.set_description(f'Computing features for shape {shape.id}')
        with ProcessPoolExecutor(max_workers=8) as executor:
            feats = executor.map(compute_features, rend_paths)
        feats = np.vstack(feats).astype(dtype=np.float32)

        parsed_filenames = [parse_rend_filename(f.name) for f in rend_paths]
        fovs = [float(d['fov']) for d in parsed_filenames]
        thetas = [float(d['theta']) for d in parsed_filenames]
        phis = [float(d['phi']) for d in parsed_filenames]

        data = {
            'fovs': np.array(fovs, dtype=np.float16),
            'thetas': np.array(thetas, dtype=np.float16),
            'phis': np.array(phis, dtype=np.float16),
            'feats': feats
        }
        shape.save_data(config.SHAPE_ALIGN_DATA_NAME, data)


if __name__ == '__main__':
    main()
