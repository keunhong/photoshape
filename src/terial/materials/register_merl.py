import collections

import click
import structlog
import csv

from terial import models
from terial.database import session_scope


logger = structlog.get_logger()


CSV_FILE_PATH = '/home/kpar/data/merl.csv'


MERLTuple = collections.namedtuple('MERLTuple', [
    'name',
    'diff_r', 'diff_g', 'diff_b',
    'spec_r', 'spec_g', 'spec_b',
    'shininess',
])


@click.command()
def main():

    with session_scope(commit=True) as sess:
        with open(CSV_FILE_PATH, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                tup = MERLTuple(*line[:len(MERLTuple._fields)])
                if tup.name.strip() == '':
                    continue
                material = models.Material(
                    type=models.MaterialType.BLINN_PHONG,
                    name=tup.name.strip(),
                    spatially_varying=False,
                    enabled=False,
                    params={
                        'diffuse': (tup.diff_r, tup.diff_g, tup.diff_b),
                        'specular': (tup.spec_r, tup.spec_g, tup.spec_b),
                        'shininess': tup.shininess,

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


if __name__ == '__main__':
    main()
