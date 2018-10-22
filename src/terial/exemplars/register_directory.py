from datetime import datetime
from pathlib import Path

import click
from skimage.color import rgb2gray
from skimage.transform import resize

import numpy as np
from skimage.io import imread, imsave
import visdom
from terial.models import Exemplar
from toolbox.masks import mask_to_bbox, crop_bbox
from terial import config, database

vis = visdom.Visdom(env='register-exemplars')


@click.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--source-name', type=str, required=True)
@click.option('--category', type=str, required=True)
def main(directory, source_name, category):
    directory = Path(directory).resolve()

    with database.session_scope() as sess:
        for path in sorted(directory.iterdir()):
            image = imread(str(path))
            cropped_image = make_square_image(image, max_size=1000)
            vis.image(cropped_image.transpose((2, 0, 1)), win='cropped')

            data_dir = Path(config.BLOB_ROOT, 'exemplars')

            if sess.query(Exemplar).filter_by(source_path=str(path)).count() > 0:
                print(f'Exemplar with path {path} already exists.')
                return

            exemplar = Exemplar(name=path.name,
                                source_path=str(path),
                                source_name=source_name,
                                date_added=datetime.utcnow(),
                                category=category)
            sess.add(exemplar)
            sess.flush()

            exemplar_dir = data_dir / str(exemplar.id)
            exemplar_dir.mkdir(parents=True, exist_ok=False)
            original_path = exemplar_dir / 'original.jpg'
            cropped_path = exemplar_dir / 'cropped.jpg'
            try:
                imsave(str(original_path), image)
                imsave(str(cropped_path), cropped_image)
            except:
                original_path.unlink()
                cropped_path.unlink()
                exemplar_dir.rmdir()
                raise

            sess.commit()
            print(f'Registered exemplars {exemplar.id}')


def bright_pixel_mask(image, percentile=80):
    image = rgb2gray(image)
    perc = np.percentile(np.unique(image), percentile)
    mask = image < perc
    return mask


def bright_pixel_bbox(image, percentile=80):
    bbox = mask_to_bbox(bright_pixel_mask(image, percentile))
    return bbox


def make_square_image(image, max_size):
    fg_bbox = bright_pixel_bbox(image, percentile=80)

    cropped_im = crop_bbox(image, fg_bbox)
    output_im = np.full((max_size, max_size, 3), dtype=cropped_im.dtype,
                        fill_value=255)
    height, width = cropped_im.shape[:2]
    if height >= width:
        new_width = int(width * max_size / height)
        padding = (max_size - new_width) // 2
        cropped_im = resize(cropped_im, (max_size, new_width),
                            mode='constant', cval=255.0, anti_aliasing=True) * 255
        output_im[:, padding:padding + new_width] = cropped_im[:, :, :3]
    else:
        new_height = int(height * max_size / width)
        padding = (max_size - new_height) // 2
        cropped_im = resize(cropped_im, (new_height, max_size),
                            mode='constant', cval=255.0, anti_aliasing=True) * 255
        output_im[padding:padding + new_height, :] = cropped_im[:, :, :3]

    return output_im


if __name__ == '__main__':
    main()
