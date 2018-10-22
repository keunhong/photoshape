import json

import click
import structlog

from terial import models, config
from terial.database import session_scope


logger = structlog.get_logger()


@click.command()
def main():

    with session_scope(commit=False) as sess:
        substance_dirs = list(config.MATERIAL_DIR_VRAY.iterdir())
        for substance_dir in substance_dirs:
            for material_dir in sorted(substance_dir.iterdir()):
                name = material_dir.name
                pre, post = name.split('_by_')
                if '-' in post:
                    subname = post.split('-')[1]
                    name = f'{pre} ({subname})'
                    author = post.split('-')[0]
                else:
                    name = pre
                    author = post

                material = models.Material(
                    type=models.MaterialType.VRAY,
                    name=name,
                    author=author,
                    substance=substance_dir.name,
                    spatially_varying=True,
                    params={
                        'raw_name': material_dir.name,
                    }
                )

                print(f'{name}, {author}')

                if (sess.query(models.Material)
                        .filter_by(type=material.type, name=material.name,
                                   author=material.author)
                        .count() > 0):
                    logger.info('Material already exists',
                                **material.serialize())
                    continue

                logger.info('Adding material', **material.serialize())

                sess.add(material)
                sess.commit()



if __name__ == '__main__':
    main()
