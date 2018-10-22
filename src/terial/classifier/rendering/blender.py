import bpy
import math
import shutil

import tempfile

from click import Path

import brender
from brender.material import DiffuseMaterial
from brender.mesh import Mesh, Plane
from brender.scene import BackgroundMode
from brender.utils import suppress_stdout
from meshkit import wavefront
from terial import models
from terial.materials import loader
from toolbox import cameras


_TMP_MESH_PATH = '/tmp/test.obj'


def construct_inference_scene(app: brender.Brender,
                              pair: models.ExemplarShapePair,
                              pair_inference_dict,
                              mat_by_id,
                              envmap: models.Envmap,
                              scene_type='inferred',
                              num_samples=256,
                              rend_shape=(1280, 1280),
                              tile_size=(512, 512),
                              frontal_camera=False,
                              diagonal_camera=False,
                              add_floor=True):
    if scene_type not in {'inferred', 'mtl'}:
        raise ValueError('Invalid scene type.')

    inference_dict = pair_inference_dict['segments']
    rk_mesh, _ = pair.shape.load(size=1)
    rk_mesh.resize(1)

    scene = brender.Scene(app, shape=rend_shape,
                          num_samples=num_samples,
                          tile_size=tile_size,
                          background_mode=BackgroundMode.COLOR,
                          background_color=(1.0, 1.0, 1.0, 0))
    envmap_rotation = (0, 0, (envmap.azimuth + math.pi/2 + pair.azimuth))
    scene.set_envmap(envmap.get_data_path('hdr.exr'),
                     scale=0.8, rotation=envmap_rotation)

    if frontal_camera:
        distance = 1.5
        fov = 50
        azimuth, elevation = pair.shape.get_frontal_angles()
    elif diagonal_camera:
        distance = 1.5
        fov = 50
        azimuth, elevation = pair.shape.get_demo_angles()
    else:
        distance = 4.0
        fov = pair.fov
        azimuth, elevation = pair.azimuth, pair.elevation

    # Get exemplar camera parameters.
    rk_camera = cameras.spherical_coord_to_cam(
        fov, azimuth, elevation, cam_dist=distance,
        max_len=rend_shape[0]/2)

    camera = brender.CalibratedCamera(scene, rk_camera.cam_to_world(), fov)

    scene.set_active_camera(camera)

    with suppress_stdout():
        mesh = Mesh.from_obj(scene, pair.shape.resized_obj_path)
        mesh.make_normals_consistent()
        mesh.enable_smooth_shading()
    mesh.recenter()

    if add_floor:
        min_pos = mesh.compute_min_pos()
        floor_mat = DiffuseMaterial(diffuse_color=(1.0, 1.0, 1.0))
        floor_mesh = Plane(position=(0, 0, min_pos))
        floor_mesh.set_material(floor_mat)

    if scene_type == 'inferred':
        for seg_id, seg_name in enumerate(rk_mesh.materials):
            if str(seg_id) not in inference_dict:
                continue
            mat_id = int(inference_dict[str(seg_id)]['material'][0]['id'])
            material = mat_by_id[mat_id]
            uv_ref_scale = 2 ** (material.default_scale - 3)
            print(f'[Pair {pair.id}] Settings segment {seg_id} ({seg_name}) '
                  f'to material {material.name}')
            # Activate only current material.
            for bobj in bpy.data.materials:
                if bobj.name == seg_name:
                    bmat = loader.material_to_brender(
                        material, bobj=bobj, uv_ref_scale=uv_ref_scale)
                    scene.add_bmat(bmat)

        # This needs to come after the materials are initialized.
        print('Computing UV density...')
        mesh.compute_uv_density()

    return scene


def animate_scene(scene: brender.Scene):
    camera = scene.camera
    camera_base_x = camera.bobj.location[0]
    camera_base_z = camera.bobj.location[2]
    camera_dists = [
        0.9, 0.8, 0.5, 0.4, 0.6, 0.7, 0.9
    ]
    camera_z = [
        0, 0.8, 1.2, 0.5, -0.1, -0.2, 0.0
    ]
    camera.track_to()

    num_orbits = 1
    frames_per_orbit = 400
    scene.bobj.frame_end = frames_per_orbit * num_orbits

    scene.bobj.frame_set(0)
    camera.camera_empty.bobj.keyframe_insert(
        data_path="rotation_euler", index=-1)
    camera.bobj.keyframe_insert(
        data_path="constraints[\"Limit Distance\"].distance", index=-1)

    for i in range(6*num_orbits):
        scene.bobj.frame_set(i * frames_per_orbit/6)
        camera.camera_empty.set_rotation((0, 0, i * 2*math.pi/3))
        camera.camera_empty.bobj.keyframe_insert(
            data_path="rotation_euler", index=-1)

    for i in range(6 * num_orbits):
        scene.bobj.frame_set(i * frames_per_orbit/6)
        r = camera_dists[i % len(camera_dists)] * camera.base_dist
        print(r)
        camera.bobj.location[2] = (
                camera_base_z + camera_z[i % len(camera_z)])
        camera.bobj.keyframe_insert(
            data_path="location", index=-1)
        camera.set_distance(camera_dists[i % len(camera_dists)])
        camera.bobj.keyframe_insert(
            data_path="constraints[\"Limit Distance\"].distance", index=-1)
        scene.bobj.frame_set(0)
