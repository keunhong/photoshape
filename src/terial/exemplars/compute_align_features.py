import argparse

from skimage.io import imread
from tqdm import tqdm

from terial import config, alignment
from terial.database import session_scope
from terial.models import Exemplar


parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
args = parser.parse_args()


def main():
    with session_scope(commit=False) as sess:
        exemplars = sess.query(Exemplar).filter_by(category=args.category).all()

    pbar = tqdm(exemplars)
    for exemplar in pbar:
        pbar.set_description(f'{exemplar.id}')
        if exemplar.data_exists(config.EXEMPLAR_ALIGN_DATA_NAME):
            tqdm.write(f'Data already exists for {exemplar.id}')
            continue
        image = imread(exemplar.cropped_path)
        feature = alignment.compute_features(
            image,
            bin_size=config.ALIGN_BIN_SIZE,
            im_shape=config.ALIGN_IM_SHAPE)
        exemplar.save_data(config.EXEMPLAR_ALIGN_DATA_NAME, feature)


if __name__ == '__main__':
    main()
