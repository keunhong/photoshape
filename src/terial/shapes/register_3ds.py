"""
Registers shapes in the 3DS Max format given a directory.
"""
import argparse
from pathlib import Path

import bpy
import shutil

from tqdm import tqdm

import brender
from brender.utils import suppress_stdout
from terial.models import Shape
from terial import config, database


TMP_OBJ_PATH = Path('/tmp/__model_uvmapped.obj')
TMP_MTL_PATH = Path('/tmp/__model_uvmapped.mtl')


parser = argparse.ArgumentParser()
parser.add_argument(dest='models_dir', type=Path)
parser.add_argument(
    '--category', type=str, required=True,
    help='The object category of the shapes being registered e.g., \'chair\'')
parser.add_argument(
    '--source-name', type=str, required=True,
    help='The name to be used as the source name e.g., \'hermanmiller\'')
args = parser.parse_args()


def register_shape(app, path):
    with suppress_stdout():
        scene = brender.Scene(app, shape=(1000, 1000))
        mesh = brender.Mesh.from_3ds(scene, path)
        mesh.remove_doubles()
        mesh.make_normals_consistent()
        mesh.enable_smooth_shading()
        mesh.unwrap_uv()

    with database.session_scope() as sess:
        shape = Shape(source=args.source_name,
                      source_id=path.name,
                      category=args.category)
        sess.add(shape)
        sess.flush()
        shape_dir = Path(config.BLOB_ROOT, 'shapes', str(shape.id))

        bpy.ops.export_scene.obj(filepath=str(TMP_OBJ_PATH))

        try:
            shape.models_dir.mkdir(parents=True)
            shutil.copy(str(TMP_OBJ_PATH), str(shape.obj_path))
            shutil.copy(str(TMP_MTL_PATH), str(shape.mtl_path))
        except:
            shape.obj_path.unlink()
            shape.mtl_path.unlink()
            shape_dir.rmdir()
            raise

        sess.commit()


def main():
    app = brender.Brender()

    paths = list(args.models_dir.glob('*/*.3DS'))
    paths.extend(args.models_dir.glob('*/*.3ds'))

    pbar = tqdm(paths)
    for hm_path in pbar:
        pbar.set_description(f'{hm_path}')

        app.init()
        try:
            register_shape(app, hm_path)
        except Exception:
            raise


if __name__ == '__main__':
    main()
