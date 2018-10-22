import click
import skimage.transform
import skimage.io
import visdom
from PIL import Image
from tqdm import tqdm

from terial import models
from terial.database import session_scope

vis = visdom.Visdom(env='resize-renderings')


SHAPES = [(384, 384), (500, 500)]


@click.command()
def main():

    with session_scope() as sess:
        rends = sess.query(models.Rendering).order_by('id').all()

    pbar = tqdm(rends)
    for rend in pbar:
        for shape in SHAPES:
            png_path = rend.get_ldr_path(shape, fmt='.png')
            jpg_path = rend.get_ldr_path(shape, fmt='.jpg')
            if not png_path.exists() or jpg_path.exists():
                continue
            pbar.set_description(f'{rend.client}/{rend.epoch}/{rend.prefix}')
            ldr = skimage.io.imread(png_path)[:, :, :3]
            im = Image.fromarray(ldr.astype('uint8'), 'RGB')
            im.save(jpg_path, quality=90)


if __name__ == '__main__':
    main()