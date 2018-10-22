import csv
import json

import click
import structlog

from brender.material import PrincipledMaterial
from terial import models, config
from terial.database import session_scope


logger = structlog.get_logger()

CSV_PATH = config.MATERIAL_DIR / 'manual_principled_brdfs.csv'


@click.command()
def main():
    with open (CSV_PATH, 'r') as f:
        with session_scope(commit=False) as sess:
            reader = csv.DictReader(f)
            for row in reader:
                material = models.Material(
                    type=models.MaterialType.PRINCIPLED,
                    name=row['name'],
                    author='Keunhong Park',
                    substance=row['substance'],
                    spatially_varying=False,
                    params={
                        'diffuse_color': eval(row['diffuse_color']),
                        'specular': float(row['specular']),
                        'metallic': float(row['metallic']),
                        'roughness': float(row['roughness']),
                        'anisotropy': float(row['anisotropy']),
                        'anisotropic_rotation': float(row['anisotropic_rotation']),
                        'clearcoat': float(row['clearcoat']),
                        'clearcoat_roughness': float(row['clearcoat_roughness']),
                        'ior': float(row['ior']),
                    }
                )

                if (sess.query(models.Material)
                        .filter_by(type=material.type, name=material.name)
                        .count() > 0):
                    logger.info('Material already exists',
                                **material.serialize())
                    continue

                logger.info('Adding material', **material.serialize())

                sess.add(material)
                sess.commit()



if __name__ == '__main__':
    main()
