from pathlib import Path

import click
import math

import numpy as np
from tqdm import tqdm
from visdom import Visdom
import brender
import brender.mesh
from brender.utils import suppress_stdout
import toolbox.images
from terial import models
from terial.materials.loader import material_to_brender
from terial.database import session_scope
from toolbox import cameras

vis = Visdom()


_TMP_MESH_PATH = '/tmp/_temp_mesh.obj'
_REND_SHAPE = (500, 500)
# _ENVMAP_PATH = Path('/home/kpar/data/envmaps/debevec/grace.pano.exr')
# _ENVMAP_PATH = Path('/home/kpar/data/envmaps/debevec/uffizi.pano.exr')
# _ENVMAP_PATH = Path('/home/kpar/data/envmaps/studioatm/StudioAtm_01_xxl.hdr')
# _ENVMAP_PATH = Path('/home/kpar/data/envmaps/custom/k_studio_dimmer.pano.exr')
_ENVMAP_PATH = Path('/home/kpar/data/envmaps/custom/studio021.hdr')


def to_8bit(image):
    return np.clip(image * 255, 0, 255).astype('uint8')


def to_srgb(linear):
    srgb = linear.copy()
    less = linear <= 0.0031308
    srgb[less] = linear[less] * 12.92
    srgb[~less] = 1.055 * np.power(linear[~less], 1.0 / 2.4) - 0.055
    return srgb


@click.command()
def main():
    with session_scope() as sess:
        materials = (sess.query(models.Material)
                     .order_by(models.Material.id.asc())
                     .all())

    # Initialize Brender and Scene.
    app = brender.Brender()
    app.init()
    scene = brender.Scene(app, shape=_REND_SHAPE, aa_samples=196)
    scene.set_envmap(_ENVMAP_PATH, scale=5.0)

    # Initialize Camera.
    rk_camera = cameras.spherical_coord_to_cam(
        60.0,
        azimuth=math.pi/2 - math.pi/12,
        elevation=math.pi/2 - math.pi/6,
        cam_dist=2.5, max_len=_REND_SHAPE[0]/2)
    camera = brender.CalibratedCamera(
        scene, rk_camera.cam_to_world(), rk_camera.fov)
    scene.set_active_camera(camera)

    with scene.select():
        mesh = brender.mesh.Monkey(position=(0, 0, 0))
        mesh.enable_smooth_shading()

    pbar = tqdm(materials)
    for material in pbar:
        pbar.set_description(material.name)
        uv_ref_scale = 2 ** (material.default_scale - 4)
        bmat = material_to_brender(material, uv_ref_scale=uv_ref_scale)
        brender.mesh.set_material(mesh.bobj, bmat)
        if bmat.has_uvs:
            mesh.compute_uv_density()

        with suppress_stdout():
            rend = scene.render_to_array(format='exr')

        material.save_data('previews/monkey.studio021.exr', rend)
        material.save_data('previews/monkey.studio021.png',
                           to_8bit(toolbox.images.linear_to_srgb(rend)))


if __name__ == '__main__':
    main()

