"""
Register ShapeNet shapes to our database given the CSV downloaded from the
ShapeNet Portal.
"""
import argparse
import json
import logging
import re
from pathlib import Path

import shutil

import csv
import visdom
from tqdm import tqdm

from terial.models import Shape
from terial import config, database

vis = visdom.Visdom()
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


pattern1 = r'(\/projects.+images\/(.*))\n'
pattern2 = r'(untitled\/(.*))\n'


shapenetcore_dir = config.SHAPENET_CORE_DIR
taxonomy_path = config.SHAPENET_TAXONOMY_PATH


parser = argparse.ArgumentParser()
parser.add_argument(dest='csv_path', type=Path)
parser.add_argument('--category', type=str, required=True)
args = parser.parse_args()


def register_shape(shapenet_path):
    shapenet_id = shapenet_path.name
    with database.session_scope() as sess:
        existing = sess.query(Shape).filter_by(source='shapenet', source_id=shapenet_id).first()
        if existing is not None:
            print(f'Shape {shapenet_id} already registered')
            return

        shape = Shape(source='shapenet',
                      source_id=shapenet_id,
                      category=args.category)
        sess.add(shape)
        sess.flush()
        shape_dir = Path(config.BLOB_ROOT, 'shapes', str(shape.id))

        original_sn_obj_path = Path(shapenet_path, 'models',
                                    'model_normalized.obj')
        original_sn_mtl_path = Path(shapenet_path, 'models',
                                    'model_normalized.mtl')
        original_obj_path = Path(shape_dir, 'models', 'original.obj')
        original_mtl_path = Path(shape_dir, 'models', 'original.mtl')

        uvmapped_sn_obj_path = Path(shapenet_path, 'models',
                                    'model_processed_v2.obj')
        uvmapped_sn_mtl_path = Path(shapenet_path, 'models',
                                    'model_processed_v2.mtl')

        if not uvmapped_sn_obj_path.exists():
            tqdm.write(f'{uvmapped_sn_obj_path} does not exist')
            return

        models_dir = shape_dir / 'models'
        try:
            models_dir.mkdir(parents=True)
        except FileExistsError:
            print(f'Removing {shape_dir}')
            shutil.rmtree(shape_dir)
            models_dir.mkdir(parents=True)

        uvmapped_obj_path = Path(shape_dir, 'models', 'uvmapped_v2.obj')
        uvmapped_mtl_path = Path(shape_dir, 'models', 'uvmapped_v2.mtl')

        sn_tex_dir = None
        tex_pattern = None
        sn_tex_dir1 = shapenet_path / 'images'
        sn_tex_dir2 = shapenet_path / 'models' / 'untitled'
        if sn_tex_dir1.exists():
            sn_tex_dir = sn_tex_dir1
            tex_pattern = pattern1
        elif sn_tex_dir2.exists():
            sn_tex_dir = sn_tex_dir2
            tex_pattern = pattern2

        if sn_tex_dir:
            if not shape.textures_dir.exists():
                shape.textures_dir.mkdir()

            for tex_path in sn_tex_dir.iterdir():
                tqdm.write(
                    f'COPY {tex_path!s} -> {shape.textures_dir / tex_path.name!s}')
                shutil.copy2(tex_path, shape.textures_dir / tex_path.name)

            with uvmapped_sn_mtl_path.open('r') as f:
                mtl_content = f.read()
                m = re.findall(tex_pattern, mtl_content)
                new_content = mtl_content
                for orig_path, tex_name in m:
                    new_path = f'../textures/{tex_name}'
                    new_content = new_content.replace(orig_path, new_path)

            with uvmapped_mtl_path.open('w') as f:
                f.write(new_content)
        else:
            tqdm.write(f'COPY {uvmapped_sn_mtl_path!s} -> {uvmapped_mtl_path!s}')
            shutil.copy2(uvmapped_sn_mtl_path, uvmapped_mtl_path)

        try:
            shutil.copy(str(original_sn_obj_path), str(original_obj_path))
            shutil.copy(str(original_sn_mtl_path), str(original_mtl_path))
            shutil.copy(str(uvmapped_sn_obj_path), str(uvmapped_obj_path))
        except:
            original_obj_path.unlink()
            original_mtl_path.unlink()
            uvmapped_obj_path.unlink()
            uvmapped_mtl_path.unlink()
            shape_dir.rmdir()
            raise

        sess.commit()

def find_shape_dir(row, parent_dict):
    shape_id = row['fullId'].split('.')[-1]
    synset_ids = row['wnsynset']
    for synset_id in synset_ids.split(','):
        parent_id = synset_id.strip()
        while parent_id in parent_dict:
            parent_id = parent_dict[parent_id]
        shape_dir = shapenetcore_dir / parent_id / shape_id
        if shape_dir.exists():
            return shape_dir
    return None


def main():
    with open(taxonomy_path, 'r') as f:
        taxonomy = json.load(f)

    parent_dict = {}
    name_dict = {}

    for synset_dict in taxonomy:
        name_dict[synset_dict['synsetId']] = synset_dict['name'].split(',')[0]
        for child in synset_dict['children']:
            parent_dict[child] = synset_dict['synsetId']

    with open(args.csv_path, 'r') as f:
        for row in csv.DictReader(f):
            shape_dir = find_shape_dir(row, parent_dict)
            register_shape(shape_dir)


if __name__ == '__main__':
    main()
