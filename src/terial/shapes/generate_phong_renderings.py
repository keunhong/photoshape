import argparse
import math
import itertools

import visdom
from tqdm import tqdm

from terial import models, config
from terial.database import session_scope
from toolbox.images import crop_tight_fg, mask_bbox
from toolbox.logging import init_logger
from vispy import app
import numpy as np
from meshkit import wavefront
from rendkit import jsd
from rendkit.materials import BlinnPhongMaterial
from rendkit.renderers import SceneRenderer
from rendkit.camera import PerspectiveCamera
from rendkit.scene import Scene

app.use_app('glfw')

vis = visdom.Visdom(env='shape-phong-rends')


logger = init_logger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--start', dest='start', type=int, default=0)
parser.add_argument('--end', dest='end', type=int, default=-1)
parser.add_argument('--preview', action='store_true')
args = parser.parse_args()


def spherical_to_cartesian(radius, azimuth, elevation):
    x = radius * math.cos(azimuth) * math.sin(elevation)
    y = radius * math.cos(elevation)
    z = radius * math.sin(azimuth) * math.sin(elevation)
    return (x, y, z)


def main():
    logger.info("Loading models.")
    with session_scope() as sess:
        shapes = (sess.query(models.Shape).order_by(models.Shape.id.asc())
                  # .filter_by(source='hermanmiller')
                  .all())

    radmap_path = '/projects/grail/kpar/data/envmaps/rnl_cross.pfm'
    radmap = jsd.import_radiance_map(dict(path=radmap_path))
    scene = Scene()
    scene.set_radiance_map(radmap)

    dist = 200
    camera = PerspectiveCamera(
        size=(1000, 1000), fov=0, near=0.1, far=5000.0,
        position=(0, 0, -dist), clear_color=(1, 1, 1, 0),
        lookat=(0, 0, 0), up=(0, 1, 0))

    renderer = SceneRenderer(scene, camera=camera,
                             gamma=2.2, ssaa=3,
                             tonemap='reinhard',
                             reinhard_thres=3.0,
                             show=args.preview)
    renderer.__enter__()

    for i, shape in enumerate(shapes):
        if shape.data_exists(config.SHAPE_REND_PHONG_NAME):
            continue
        if i < args.start:
            continue
        if args.end != -1 and i >= args.end:
            logger.info("Hit end {}={}.".format(i, args.end))
            return

        logger.info("[%d/%d] Processing %d", i + 1, len(shapes), shape.id)

        print(shape.obj_path)
        if not shape.obj_path.exists():
            logger.error('Shape %d does not have a UV mapped model.', shape.id)
            continue

        mesh = wavefront.read_obj_file(shape.obj_path)
        mesh.resize(100)
        materials = wavefront.read_mtl_file(shape.mtl_path, mesh)
        scene.add_mesh(mesh)

        for mat_name, mat in materials.items():
            roughness = math.sqrt(2/(mat.specular_exponent + 2))
            # material = BlinnPhongMaterial(mat.diffuse_color,
            #                               mat.specular_color,
            #                               roughness)
            material = BlinnPhongMaterial(diff_color=(0.1, 0.28, 0.8),
                                          spec_color=(0.04, 0.04, 0.04),
                                          roughness=0.1)
            scene.put_material(mat_name, material)

        data_name = f'preview/phong.500x500.png'
        # if shape.data_exists(data_name):
        #     continue

        camera.position = spherical_to_cartesian(dist, *shape.get_demo_angles())
        camera.fov = 50
        if args.preview:
            renderer.draw()
            renderer.swap_buffers()
         # Set format to RGBA so we can crop foreground using alpha.
        image = renderer.render_to_image(format='rgba')
        image = np.clip(image, 0, 1)
        fg_bbox = mask_bbox(image[:, :, 3] > 0)
        image = crop_tight_fg(
            image[:, :, :3], (500, 500), bbox=fg_bbox, use_pil=True)
        vis.image(image.transpose((2, 0, 1)), win='rend-preview')

        shape.save_data(config.SHAPE_REND_PHONG_NAME, image)
        scene.clear()


if __name__ == '__main__':
    main()
