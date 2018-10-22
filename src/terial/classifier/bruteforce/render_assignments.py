import math
from pathlib import Path

import bpy
import click
import logging

import json
import visdom

import toolbox.images
import brender
import toolbox.io
import toolbox.io.images
from brender import Mesh
from brender.material import DiffuseMaterial
from brender.mesh import Plane
from brender.utils import suppress_stdout
from meshkit import wavefront
from rendkit.shortcuts import render_segments
import vispy.app
from terial import config, models, controllers
from terial.materials import loader
from terial.database import session_scope
from terial.models import ExemplarShapePair
from toolbox import cameras

vispy.app.use_app('glfw')
vis = visdom.Visdom(env='brdf-classifier-baseline-color')


logger = logging.getLogger(__name__)


_TMP_MESH_PATH = '/tmp/_temp_mesh.obj'
_TMP_REND_PATH = '/tmp/_temp_rend.png'
# _ENVMAP_PATH = '/projects/grail/kpar/data/envmaps2/k_studio_dimmer.pano.exr'
# _ENVMAP_PATH = '/projects/grail/kpar/data/envmaps2/uffizi.pano.exr'
_ENVMAP_PATH = '/local1/data/envmaps/custom/studio021.hdr'
_REND_SHAPE = (768, 768)
_FINAL_SHAPE = (500, 500)


# MAT_CAND_NAME = config.MATERIAL_CAND_VGG16_WHOLE_NAME
MAT_CAND_NAME = config.MATERIAL_CAND_MEDCOL_NAME
# MAT_CAND_NAME = config.MATERIAL_CAND_VGG16_PATCH_NAME


def render_pair(app: brender.Brender, pair: ExemplarShapePair,
                seg_mat_candidates):
    rk_mesh, _ = pair.shape.load()
    # Load shapenet mesh and resize to 1.0 to match Blender size.
    rk_mesh.resize(1)
    with open(_TMP_MESH_PATH, 'w') as f:
        wavefront.save_obj_file(f, rk_mesh)

    scene = brender.Scene(app, shape=_REND_SHAPE,
                          tile_size=(40, 40),
                          aa_samples=96,
                          diffuse_samples=3,
                          specular_samples=3)
    envmap_rotation = (0, 0, (math.pi + math.pi/2 + pair.azimuth))
    scene.set_envmap(_ENVMAP_PATH, scale=5, rotation=envmap_rotation)

    # Get exemplar camera parameters.
    rk_camera = cameras.spherical_coord_to_cam(
        pair.fov, pair.azimuth, pair.elevation, cam_dist=2.0,
        max_len=_REND_SHAPE[0]/2)
    segment_im = render_segments(rk_mesh, rk_camera)
    fg_bbox = toolbox.images.mask_bbox(segment_im > -1)

    camera = brender.CalibratedCamera(
        scene, rk_camera.cam_to_world(), pair.fov)
    scene.set_active_camera(camera)

    with suppress_stdout():
        mesh = Mesh.from_obj(scene, _TMP_MESH_PATH)
        mesh.enable_smooth_shading()
        mesh.recenter()
        floor_mat = DiffuseMaterial(diffuse_color=(0.3, 0.3, 0.3))
        floor_mesh = Plane(position=(0, 0, mesh.compute_min_pos()))
        floor_mesh.set_material((0, 0, floor_mat))

    bmats = []

    with session_scope() as sess:
        for seg_id, seg_name in enumerate(rk_mesh.materials):
            if seg_name not in seg_mat_candidates:
                logger.warning('[Pair %d] Segment %d (%s) is not in dict',
                               pair.id, seg_id, seg_name)
                continue
            candidate = seg_mat_candidates[seg_name][0]
            mat_id = candidate['id']
            material = sess.query(models.Material).filter_by(id=mat_id).first()
            if material is None:
                logger.error('Material with id=%d not found', mat_id)
                return None, None
            uv_ref_scale = 2 ** (material.default_scale - 3)

            print(f'Settings {seg_name} to material {mat_id}')
            # Activate only current material.
            for bobj in bpy.data.materials:
                if bobj.name == seg_name:
                    bmat = loader.material_to_brender(
                        material, bobj=bobj, uv_ref_scale=uv_ref_scale)
                    bmats.append(bmat)

    # This needs to come after the materials are initialized.
    print('Computing UV density...')
    mesh.compute_uv_density()

    # blend_path = Path(pair.get_image_dir_path(), MAT_CAND_NAME + '.blend')
    # blend_path.parent.mkdir(parents=True, exist_ok=True)
    # bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    print('Rendering...')
    with suppress_stdout():
        rend = scene.render_to_array(format='exr')

    while len(bmats) > 0:
        bmat = bmats.pop()
        bmat.bobj.name = bmat.bobj.name
        del bmat

    rend_srgb = toolbox.images.linear_to_srgb(rend)
    rend_srgb = toolbox.images.crop_tight_fg(
        rend_srgb, _FINAL_SHAPE, bbox=toolbox.images.bbox_make_square(fg_bbox),
        fill=255, use_pil=True)

    return rend, rend_srgb


@click.command()
@click.argument('in-dir', type=click.Path())
@click.argument('out-dir', type=click.Path())
def main(in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    app = brender.Brender()

    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs(
            sess,
            by_shape=True,
            max_dist=config.ALIGN_DIST_THRES,
            order_by=ExemplarShapePair.distance.asc(),
        )
        # pairs: List[ExemplarShapePair] = (
        #     sess.query(ExemplarShapePair)
        #         .filter(ExemplarShapePair.distance < config.ALIGN_DIST_THRES)
        #         .options(orm.joinedload(models.ExemplarShapePair.exemplar),
        #                  orm.joinedload(models.ExemplarShapePair.shape))
        #         .order_by(ExemplarShapePair.id.asc())
        #         .all())

        print(f'Fetched {len(pairs)} pairs. '
              f'align_dist_thres = {config.ALIGN_DIST_THRES}')

        for pair in pairs:

            cand_path = in_dir / f'{pair.id}.json'
            if not cand_path.exists():
                continue

            with cand_path.open('r') as f:
                seg_cands = json.load(f)

            print(f'Processing pair {pair.id}')
            app.init()

            hdr_path = out_dir / f'{pair.id:04d}.exr'
            ldr_path = out_dir / f'{pair.id:04d}.png'

            if ldr_path.exists():
                continue

            try:
                rend, rend_srgb = render_pair(app, pair, seg_cands)
            except Exception as e:
                logger.exception('Caught exception')
                continue

            if rend is None:
                continue
            vis.image(rend_srgb.transpose((2, 0, 1)))

            toolbox.io.images.save_hdr(hdr_path, rend)
            toolbox.io.images.save_image(ldr_path, rend_srgb)


if __name__ == '__main__':
    main()


