import json

import click
import structlog

from terial import models, config
from terial.database import session_scope


logger = structlog.get_logger()


@click.command()
def main():

    with session_scope(commit=True) as sess:
        substance_dirs = list(config.MATERIAL_DIR_AITTALA.iterdir())
        for substance_dir in substance_dirs:
            for material_dir in substance_dir.iterdir():
                material = models.Material(
                    type=models.MaterialType.AITTALA_BECKMANN,
                    name=material_dir.name,
                    substance=substance_dir.name,
                    spatially_varying=True,
                )

                annot_path = material_dir / 'annotations.json'
                if annot_path.exists():
                    with open(annot_path, 'r') as f:
                        annot = json.load(f)
                        if 'scale' in annot:
                            material.min_scale = annot['scale']

                if (sess.query(models.Material)
                        .filter_by(type=material.type, name=material.name)
                        .count() > 0):
                    logger.info('Material already exists',
                                **material.serialize())
                    continue

                logger.info('Adding material', **material.serialize())

                sess.add(material)



if __name__ == '__main__':
    main()
