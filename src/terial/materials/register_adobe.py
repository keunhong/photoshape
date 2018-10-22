import argparse
from pathlib import Path

import structlog

import mdl
from terial import models
from terial.database import session_scope


logger = structlog.get_logger()


parser = argparse.ArgumentParser()
parser.add_argument(dest='base_dir', type=str)
args = parser.parse_args()


def main():

    with session_scope(commit=False) as sess:
        substance_dirs = list(Path(args.base_dir).iterdir())
        for substance_dir in sorted(substance_dirs):
            mdl_paths = []
            for material_dir in substance_dir.iterdir():
                mdl_path = list(material_dir.glob('*.mdl'))[0]
                mdl_paths.append(mdl_path)

            for mdl_path in sorted(mdl_paths, key=lambda s: s.stem):
                mdl_dict = mdl.parse_mdl(mdl_path)
                spatially_varying = isinstance(mdl_dict['base_color'], str)

                material = models.Material(
                    type=models.MaterialType.MDL,
                    name=mdl_path.stem,
                    substance=substance_dir.name,
                    spatially_varying=spatially_varying,
                    source='adobe_stock',
                    source_id=mdl_path.parent.name.split('_')[1],
                    params=mdl_dict,
                )

                if (sess.query(models.Material)
                        .filter_by(type=material.type,
                                   source_id=material.source_id)
                        .count() > 0):
                    logger.info('Material already exists',
                                **material.serialize())
                    continue

                logger.info('Adding material', **material.serialize())

                sess.add(material)
                sess.commit()



if __name__ == '__main__':
    main()
