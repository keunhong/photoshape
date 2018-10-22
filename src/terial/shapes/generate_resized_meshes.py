import re
import shutil
from pathlib import Path

from tqdm import tqdm

from meshkit import wavefront
from terial import models
from terial.database import session_scope


def main():
    with session_scope() as sess:
        shapes = sess.query(models.Shape).all()

    pbar = tqdm(shapes)
    shape: models.Shape
    for shape in pbar:
        pbar.set_description(f"{shape.id!s}")

        new_obj_path = shape.models_dir / 'uvmapped_v2.resized.obj'
        new_mtl_path = shape.models_dir / 'uvmapped_v2.resized.mtl'
        if new_obj_path.exists():
            continue

        mesh = wavefront.read_obj_file(shape.obj_path)
        mesh.resize(1.0)
        with new_obj_path.open('w') as f:
            wavefront.save_obj_file(f, mesh)
        shutil.copy2(shape.mtl_path, new_mtl_path)

if __name__ == '__main__':
    main()