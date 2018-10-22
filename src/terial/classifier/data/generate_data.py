"""
Generates exemplar-shape pairs.
"""
import asyncio
import itertools

import collections
import json
import random

import bpy
from typing import List, Tuple, Dict

import numpy as np
import attr
import click as click
import math
import sqlalchemy as sa
import time
import visdom
from sqlalchemy import orm

import brender
from brender import Mesh
from brender.material import NodesMaterial
from brender.scene import BackgroundMode
from brender.utils import suppress_stdout
from meshkit import wavefront
from rendkit import shortcuts
from terial import config, models
from terial.config import SUBSTANCES
from terial.materials import loader
from terial.classifier.data import collector
from terial.models import Shape, ExemplarShapePair, Exemplar
from terial.database import session_scope
from toolbox import cameras
from terial.pairs import utils
from terial.web.utils import make_http_client
from toolbox.images import visualize_map
import toolbox.images
import vispy
from toolbox.logging import init_logger

vispy.use(app='glfw')  # noqa

logger = init_logger(__name__)

vis = visdom.Visdom(env='generate-data')


_TMP_MESH_PATH = '/tmp/_terial_generate_data_temp_mesh.obj'
_TMP_REND_PATH = '/tmp/_terial_generate_data_temp_rend.exr'
_REND_SHAPE = (500, 500)


@attr.s()
class _Envmap(object):
    path: str = attr.ib()
    rotation: Tuple[float, float, float] = attr.ib()


pi = math.pi


FOV_MIN = 50.0
FOV_MAX = 60.0


@click.command()
@click.option('--client-id', required=True, type=str)
@click.option('--epoch-start', default=0)
@click.option('--rends-per-epoch', default=1)
@click.option('--dry-run', is_flag=True)
@click.option('--max-dist', default=12.0)
@click.option('--required-substances', type=str)
@click.option('--host', required=True)
@click.option('--port', default=9600)
def main(client_id, epoch_start, rends_per_epoch, dry_run,
         max_dist, host, port, required_substances):
    loop = asyncio.get_event_loop()
    http_sess = loop.run_until_complete(make_http_client())

    app = brender.Brender()

    collector_ctx = {
        'host': host,
        'port': port,
        'sess': http_sess,
        'client_id': client_id,
    }

    if required_substances:
        required_substances = set(required_substances.split(','))
        assert all(s in SUBSTANCES for s in required_substances)

    for epoch in itertools.count(start=epoch_start):
        logger.info('Starting epoch %d', epoch)
        with session_scope() as sess:
            pairs = (
                sess.query(ExemplarShapePair)
                    .join(Shape)
                    .join(Exemplar)
                    .filter(sa.and_(
                        ExemplarShapePair.distance < max_dist,
                        sa.not_(Shape.exclude),
                        sa.not_(Exemplar.exclude),
                        Shape.split_set.isnot(None),
                    ))
                    .options(orm.joinedload(ExemplarShapePair.exemplar),
                             orm.joinedload(ExemplarShapePair.shape))
                    .order_by(Shape.id.asc())
                    .all())
            materials = sess.query(models.Material).filter_by(enabled=True).all()
            envmaps = sess.query(models.Envmap).filter_by(enabled=True).all()

        cam_angles = [(p.azimuth, p.elevation) for p in pairs]
        logger.info('Loaded %d pairs and %d camera angles', len(pairs), len(cam_angles))

        mats_by_subst = collections.defaultdict(list)
        for material in materials:
            mats_by_subst[material.substance].append(material)

        envmaps_by_split = {
            'train': [e for e in envmaps if e.split_set == 'train'],
            'validation': [e for e in envmaps if e.split_set == 'validation'],
        }

        for i in range(1000):
            pair = random.choice(pairs)
            if not pair.data_exists(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME):
                continue

            app.init()
            logger.info('pair %d, shape %s, exemplar %d', pair.id, pair.shape.id,
                        pair.exemplar.id)
            try:
                loop.run_until_complete(
                    process_pair(app,
                                 pair,
                                 cam_angles,
                                 mats_by_subst,
                                 envmaps_by_split=envmaps_by_split,
                                 num_rends=rends_per_epoch,
                                 is_dry_run=dry_run,
                                 epoch=epoch,
                                 collector_ctx=collector_ctx,
                                 required_substances=required_substances))
            except Exception as e:
                logger.exception('Uncaught exception', exc_info=True)
                continue


async def process_pair(app: brender.Brender,
                       pair: ExemplarShapePair,
                       cam_angles: List[Tuple[float, float]],
                       mats_by_subst: Dict[str, List[NodesMaterial]],
                       num_rends,
                       *,
                       envmaps_by_split,
                       is_dry_run,
                       epoch,
                       collector_ctx,
                       required_substances=None):
    rk_mesh, _ = pair.shape.load()
    rk_mesh.resize(1)
    with open(_TMP_MESH_PATH, 'w') as f:
        wavefront.save_obj_file(f, rk_mesh)

    seg_substances = utils.compute_segment_substances(pair)

    if required_substances:
        for subst in required_substances:
            if subst not in seg_substances.values():
                return

    print(seg_substances.values())

    scene = brender.Scene(app, shape=_REND_SHAPE,
                          tile_size=(40, 40),
                          aa_samples=128,
                          diffuse_samples=3,
                          specular_samples=3,
                          background_mode=BackgroundMode.COLOR,
                          background_color=(1, 1, 1, 1))

    with suppress_stdout():
        mesh = Mesh.from_obj(scene, _TMP_MESH_PATH)
        # mesh.remove_doubles()
        mesh.enable_smooth_shading()

    for i in range(num_rends):
        bmats = []

        try:
            r = do_render(scene, pair, mesh, envmaps_by_split, cam_angles,
                      rk_mesh, seg_substances, mats_by_subst, bmats)
        finally:
            while len(bmats) > 0:
                bmat = bmats.pop()
                bmat.bobj.name = bmat.bobj.name
                del bmat

        if not is_dry_run:
            await collector.send_data(
                **collector_ctx,
                split_set=pair.shape.split_set,
                pair_id=pair.id,
                epoch=epoch,
                iteration=i,
                params=r['params'],
                ldr_image=r['ldr'],
                hdr_image=r['hdr'],
                seg_map=r['seg_map'],
                seg_vis=r['seg_vis'],
                normal_image=r['normal_image'],
            )


def do_render(scene, pair, mesh, envmaps_by_split, cam_angles,
              rk_mesh, seg_substances, mats_by_subst, bmats):
    time_begin = time.time()

    # Jitter camera params.
    cam_azimuth, cam_elevation = random.choice(cam_angles)
    cam_azimuth += random.uniform(-math.pi/12, -math.pi/12)
    cam_elevation += random.uniform(-math.pi/24, -math.pi/24)
    cam_dist = random.uniform(1.3, 1.75)
    cam_fov = random.uniform(FOV_MIN, FOV_MAX)
    rk_camera = cameras.spherical_coord_to_cam(
        cam_fov, cam_azimuth, cam_elevation, cam_dist=cam_dist,
        max_len=_REND_SHAPE[0]/2)

    # Jitter envmap params.
    # Set envmap rotation so that the camera points at the "back"
    # but also allow some wiggle room.
    envmap_scale = random.uniform(0.9, 1.2)
    envmap = random.choice(envmaps_by_split[pair.shape.split_set])
    envmap_rotation = (0, 0, (envmap.azimuth + pi/2 + cam_azimuth
                              + random.uniform(-pi/24, pi/24)))
    scene.set_envmap(
        envmap.get_data_path('hdr.exr'),
        scale=envmap_scale, rotation=envmap_rotation)

    if scene.camera is None:
        camera = brender.CalibratedCamera(
            scene, rk_camera.cam_to_world(), cam_fov)
        scene.set_active_camera(camera)
    else:
        scene.camera.set_params(rk_camera.cam_to_world(), cam_fov)

    segment_material_ids = {}
    segment_uv_ref_scales = {}
    segment_uv_rotations = {}
    segment_uv_translations = {}
    segment_mean_roughness = {}

    logger.info('Setting materials...')

    brdf_names = set()
    substances = set()
    for seg_name in rk_mesh.materials:
        for bobj in bpy.data.materials:
            if bobj.name == seg_name:
                if seg_name not in seg_substances:
                    logger.warning('Substance unknown for %s', seg_name)
                    return
                substance = seg_substances[seg_name]
                substances.add(substance)
                materials = mats_by_subst[substance]
                if len(materials) == 0:
                    logger.warning('No materials for substance %s', substance)
                    return
                material: models.Material = random.choice(materials)
                brdf_names.add(material.name)

                # Jitter UV map.
                uv_translation = (random.uniform(0, 1),
                                  random.uniform(0, 1))
                uv_rotation = random.uniform(0, 2 * math.pi)
                uv_ref_scale = (2 ** (material.default_scale - 3.0
                                      + random.uniform(-1.0, 0.5)))

                segment_uv_ref_scales[seg_name] = uv_ref_scale
                segment_material_ids[seg_name] = material.id
                segment_uv_rotations[seg_name] = uv_rotation
                segment_uv_translations[seg_name] = uv_translation
                bmat: NodesMaterial = loader.material_to_brender(
                    material, bobj=bobj,
                    uv_ref_scale=uv_ref_scale,
                    uv_translation=uv_translation,
                    uv_rotation=uv_rotation)
                segment_mean_roughness[seg_name] = \
                    float(bmat.mean_roughness())

                bmats.append(bmat)

    # This needs to come after the materials are initialized.
    logger.info('Computing UV density...')
    mesh.compute_uv_density()

    logger.info('Rendering...')

    with suppress_stdout():
        rend = scene.render_to_array(format='exr')

    caption = f'{envmap.name}, {str(substances)}, {str(brdf_names)}'
    rend_srgb = toolbox.images.linear_to_srgb(rend)

    time_elapsed = time.time() - time_begin
    logger.info('Rendered one in %fs', time_elapsed)

    seg_map = shortcuts.render_segments(rk_mesh, rk_camera)
    seg_vis = visualize_map(seg_map)[:, :, :3]
    normal_map = shortcuts.render_mesh_normals(rk_mesh, rk_camera)

    normal_map_blender = normal_map.copy()
    normal_map_blender[:, :, :3] += 1.0
    normal_map_blender[:, :, :3] /= 2.0
    normal_map_blender = np.round(255.0 * normal_map_blender).astype(np.uint8)

    figure = toolbox.images.to_8bit(np.hstack((
        seg_vis, rend_srgb[:, :, :3],
    )))

    segment_ids = {
        name: i for i, name in enumerate(rk_mesh.materials)
    }

    params = {
        'split_set': pair.shape.split_set,
        'pair_id': pair.id,
        'shape_id': pair.shape_id,
        'exemplar_id': pair.exemplar_id,
        'camera': {
            'fov': cam_fov,
            'azimuth': cam_azimuth,
            'elevation': cam_elevation,
            'distance': cam_dist,
        },
        'envmap': {
            'id': envmap.id,
            'name': envmap.name,
            'source': envmap.source,
            'scale': envmap_scale,
            'rotation': envmap_rotation,
        },
        'segment': {
            'segment_ids': segment_ids,
            'materials': segment_material_ids,
            'uv_ref_scales': segment_uv_ref_scales,
            'uv_translations': segment_uv_translations,
            'uv_rotations': segment_uv_rotations,
            'mean_roughness': segment_mean_roughness,
        },
        'time_elapsed': time_elapsed,
    }

    return {
        'ldr': toolbox.images.to_8bit(rend_srgb),
        'hdr': rend,
        'seg_map': (seg_map + 1).astype(np.uint8),
        'seg_vis': toolbox.images.to_8bit(seg_vis),
        'normal_image': normal_map_blender,
        'params': params,
    }


if __name__ == '__main__':
    main()
