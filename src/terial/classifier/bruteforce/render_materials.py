import math
from collections import defaultdict

import bpy
from pathlib import Path

import click
import structlog
import visdom
import warnings
from tqdm import tqdm
from skimage import io as skio

import brender
from brender import Mesh
from brender.material import InvisibleMaterial
from brender.utils import suppress_stdout
from meshkit import wavefront
from rendkit.shortcuts import render_segments
import vispy.app
from terial import config, controllers
from terial.materials import loader
from terial.database import session_scope
from terial.models import ExemplarShapePair, Material
from toolbox import cameras
from terial.pairs import utils
from toolbox.images import crop_tight_fg, mask_bbox

logger = structlog.get_logger(__name__)

vispy.app.use_app('glfw')
vis = visdom.Visdom(env='bruteforce-render-materials')


_TMP_MESH_PATH = '/tmp/_temp_mesh.obj'
_TMP_REND_PATH = '/tmp/_temp_rend.png'
# _ENVMAP_PATH = '/projects/grail/kpar/data/envmaps2/k_studio_dimmer.pano.exr'
_ENVMAP_PATH = '/local1/data/envmaps/custom/studio021.hdr'
_REND_SHAPE = (256, 256)
_FINAL_SHAPE = (128, 128)
_BASE_NAME = 'material_rends_256x256'


def render_pair(app: brender.Brender, pair: ExemplarShapePair,
                materials_by_substance, base_out_dir):
    # Load shapenet mesh and resize to 1.0 to match Blender size.
    rk_mesh, _ = pair.shape.load()
    rk_mesh.resize(1)
    with open(_TMP_MESH_PATH, 'w') as f:
        wavefront.save_obj_file(f, rk_mesh)

    scene = brender.Scene(app, shape=_REND_SHAPE, aa_samples=32)
    envmap_rotation = (0, 0, (math.pi + math.pi/2 + pair.azimuth))
    scene.set_envmap(_ENVMAP_PATH, scale=5, rotation=envmap_rotation)

    with suppress_stdout():
        mesh = Mesh.from_obj(scene, _TMP_MESH_PATH)

    mat_substances = utils.compute_segment_substances(pair)

    # Get exemplar camera parameters.
    rk_camera = cameras.spherical_coord_to_cam(
        pair.fov, pair.azimuth, pair.elevation, cam_dist=2.0,
        max_len=_REND_SHAPE[0]/2)

    segment_im = render_segments(rk_mesh, rk_camera)

    camera = brender.CalibratedCamera(
        scene, rk_camera.cam_to_world(), pair.fov)
    scene.set_active_camera(camera)

    bmats = []
    segment_pbar = tqdm(rk_mesh.materials)
    for segment_name in segment_pbar:
        segment_pbar.set_description(f'Segment {segment_name}')

        try:
            mat_subst = mat_substances[segment_name]
            materials = materials_by_substance[mat_subst]
        except KeyError:
            continue

        out_dir = Path(base_out_dir, str(pair.id), str(segment_name))
        out_dir.mkdir(parents=True, exist_ok=True)

        material_pbar = tqdm(materials)
        for material in material_pbar:
            material_pbar.set_description(f'Material {material.id}')
            out_path = Path(out_dir, f'{material.id}.png')
            if out_path.exists():
                material_pbar.set_description(
                    f'Material {material.id} already rendered')
                continue

            # Activate only current material.
            bobj = None
            for bobj in bpy.data.materials:
                if bobj.name == segment_name:
                    break

            bobj_matches = [o for o in bpy.data.materials
                            if o.name == segment_name]
            if len(bobj_matches) == 0:
                bmat = InvisibleMaterial(bobj=bobj)
            else:
                bmat = loader.material_to_brender(material, bobj=bobj)

            bmats.append(bmat)

            with suppress_stdout():
                rend_im = scene.render_to_array()

            vis.image(rend_im.transpose((2, 0, 1)), win='rend-im')

            rend_im[segment_im != rk_mesh.materials.index(segment_name)] = 0
            fg_bbox = mask_bbox(segment_im > -1)
            rend_im = crop_tight_fg(rend_im, _FINAL_SHAPE, bbox=fg_bbox,
                                    fill=0, use_pil=True)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                skio.imsave(str(out_path), rend_im)

    while len(bmats) > 0:
        bmat = bmats.pop()
        bmat.bobj.name = bmat.bobj.name
        del bmat


@click.command()
@click.argument('out_dir', type=click.Path())
def main(out_dir):
    out_dir = Path(out_dir)
    app = brender.Brender()

    materials_by_substance = defaultdict(list)
    with session_scope() as sess:
        materials = sess.query(Material).filter_by(enabled=True).all()
        for material in materials:
            materials_by_substance[material.substance].append(material)

        # pairs, count = controllers.fetch_pairs(
        #     sess, max_dist=config.ALIGN_DIST_THRES,
        #     filters=[ExemplarShapePair.id >= start],
        #     order_by=ExemplarShapePair.shape_id.asc(),
        # )

        pairs, count = controllers.fetch_pairs(
            sess,
            by_shape=True,
            order_by=ExemplarShapePair.distance.asc(),
        )

        print(f'Fetched {len(pairs)} pairs. '
              f'align_dist_thres = {config.ALIGN_DIST_THRES}')

        pair_pbar = tqdm(pairs)
        for i, pair in enumerate(pair_pbar):
            pair_pbar.set_description(f'Pair {pair.id}')

            if not pair.data_exists(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME):
                continue

            app.init()
            render_pair(app, pair, materials_by_substance, out_dir)


if __name__ == '__main__':
    main()


