from pathlib import Path

import bpy
import click
import math

import numpy as np
from tqdm import tqdm
from visdom import Visdom
import brender
import brender.mesh
from brender.utils import suppress_stdout
import toolbox.images
from meshkit import wavefront
from terial import models
from terial.materials.loader import material_to_brender
from terial.database import session_scope
from terial.models import MaterialType
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
                     # .filter(sa.and_(models.Material.default_scale.isnot(None),
                     #                 models.Material.substance == 'fabric'))
                     # .filter(models.Material.type == MaterialType.MDL)
                     .order_by(models.Material.id.asc())
                     .all())
        shape = sess.query(models.Shape).get(4682)
        # shape = sess.query(models.Shape).get(2333)

    # Initialize Brender and Scene.
    app = brender.Brender()
    app.init()
    scene = brender.Scene(app, shape=_REND_SHAPE, aa_samples=196)
    envmap_rotation = (0, 0, (math.pi + math.pi/2 - math.pi / 2 - math.pi / 12))
    scene.set_envmap(_ENVMAP_PATH, scale=2.0, rotation=envmap_rotation)

    # Initialize Camera.
    rk_camera = cameras.spherical_coord_to_cam(
        60.0,
        azimuth=-math.pi / 2 - math.pi / 12,
        elevation=math.pi / 2 - math.pi / 6,
        cam_dist=1.2, max_len=_REND_SHAPE[0] / 2)
    camera = brender.CalibratedCamera(
        scene, rk_camera.cam_to_world(), rk_camera.fov)
    scene.set_active_camera(camera)

    # Load shapenet mesh and resize to 1.0 to match Blender size.
    rk_mesh, _ = shape.load()
    rk_mesh.resize(1)
    with open(_TMP_MESH_PATH, 'w') as f:
        wavefront.save_obj_file(f, rk_mesh)
    mesh = brender.mesh.Mesh.from_obj(scene, _TMP_MESH_PATH)

    # Align the mesh to a camera looking straight at the diffuser. The diffuser
    # for Studio 021 is at azimuth=pi/2, elevation=pi/2.
    # brender.mesh.align_mesh_to_direction(mesh, math.pi / 2, math.pi / 2)

    # with scene.select():
    #     mesh = brender.mesh.Monkey(position=(0, 0, 0))

    pbar = tqdm(materials)
    for material in pbar:
        uv_ref_scale = 2 ** (material.default_scale - 4)
        pbar.set_description(material.name)

        data_name = f'previews/chair_{material.default_scale}.png'
        if material.data_exists(data_name):
            continue

        # _, _, uv_density = measure_uv_density(mesh.bobj)
        bmat = material_to_brender(material, uv_ref_scale=uv_ref_scale)
        brender.mesh.set_material(mesh.bobj, bmat)
        if bmat.has_uvs:
            mesh.compute_uv_density()

        with suppress_stdout():
            rend = scene.render_to_array(format='exr')

        # material.save_data(
        #     f'previews/chair.exr',
        #     rend)
        bpy.ops.wm.save_as_mainfile(filepath='/local1/data/test.blend')
        material.save_data(data_name,
                           to_8bit(toolbox.images.linear_to_srgb(rend)))


if __name__ == '__main__':
    main()
