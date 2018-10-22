from pathlib import Path

import click
import toolbox.images

import visdom

import toolbox.io
import toolbox.io.images
from terial import models
from terial import config, database

vis = visdom.Visdom()


@click.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--name', type=str, required=True)
@click.option('--source', type=str, required=True)
def main(filepath, name, source):
    filepath = Path(filepath)
    try:
        hdr = toolbox.io.images.load_hdr(filepath)
    except FileNotFoundError:
        print(f"File {filepath!r} does not exist.")
        return

    ldr = toolbox.images.to_8bit(toolbox.images.linear_to_srgb(hdr))

    data_dir = Path(config.BLOB_ROOT, 'envmaps')

    with database.session_scope() as sess:
        envmap = models.Envmap(name=name, source=source)
        sess.add(envmap)
        sess.flush()

        envmap.data_path.mkdir(parents=True)
        envmap.save_data('hdr.exr', hdr)
        envmap.save_data('preview.png', ldr)

        sess.commit()
        print(f'Registered envmap {name} as {envmap.id}')


if __name__ == '__main__':
    main()
