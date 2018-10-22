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
from terial.database import session_scope
from terial.models import Shape
from terial import config, database, models

TMP_OBJ_PATH = Path('/tmp/__model_uvmapped.obj')
TMP_MTL_PATH = Path('/tmp/__model_uvmapped.mtl')


parser = argparse.ArgumentParser()
parser.add_argument(dest='models_dir', type=Path)
args = parser.parse_args()


def register_shape(app, shape, path):
    with suppress_stdout():
        scene = brender.Scene(app, shape=(1000, 1000))
        mesh = brender.Mesh.from_3ds(scene, path)
        mesh.remove_doubles()
        mesh.make_normals_consistent()
        mesh.enable_smooth_shading()
        mesh.unwrap_uv()

    shape_dir = Path(config.BLOB_ROOT, 'shapes', str(shape.id))

    bpy.ops.export_scene.obj(filepath=str(TMP_OBJ_PATH))

    try:
        if not shape.models_dir.exists():
            shape.models_dir.mkdir(parents=True)
        shutil.copy(str(TMP_OBJ_PATH), str(shape.obj_path))
        shutil.copy(str(TMP_MTL_PATH), str(shape.mtl_path))
    except:
        shape.obj_path.unlink()
        shape.mtl_path.unlink()
        shape_dir.rmdir()
        raise


def main():
    paths = []
    for ext in ('*.3ds', '*.3DS'):
        paths.extend(args.models_dir.glob(ext))

    for path in paths:
        with session_scope() as sess:
            shapes = sess.query(models.Shape).filter_by(source_id=path.name).all()
        print([s for s in shapes])

    app = brender.Brender()

    pbar = tqdm(paths)
    for path in pbar:
        pbar.set_description(f'{path}')

        with session_scope() as sess:
            shapes = sess.query(models.Shape).filter_by(source_id=path.name).all()

        if len(shapes) == 0:
            continue

        for shape in shapes:
            app.init()
            try:
                register_shape(app, shape, path)
            except Exception:
                raise


if __name__ == '__main__':
    main()
