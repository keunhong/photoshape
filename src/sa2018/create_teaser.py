import os

import numpy as np
import json

from pathlib import Path

import bpy

import math

import tempfile

import uuid

import random

from tqdm import tqdm

import brender
from brender import Mesh
from brender.material import DiffuseMaterial, BlinnPhongMaterial
from brender.mesh import Plane, select_objects
from brender.scene import BackgroundMode, Engine
from brender.utils import suppress_stdout
from meshkit import wavefront
from terial import models
from terial.materials import loader
from terial.database import session_scope
from terial.models import ExemplarShapePair
from toolbox import cameras


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest='list_path', type=Path)
parser.add_argument(dest='out_path', type=Path)
parser.add_argument('--inference-dir', type=Path)
parser.add_argument('--type', choices=['inferred', 'none', 'mtl'])
parser.add_argument('--pack-assets', action='store_true')
args = parser.parse_args()


_REND_SHAPE = (1200, 1920)


def main():
    app = brender.Brender()

    if not args.list_path.exists():
        print(f'{args.list_path!s} does not exist.')
        return

    with args.list_path.open('r') as f:
        file_list = f.read().strip().split('\n')
        inference_paths = [Path(args.inference_dir, f'{p}.json')
                           for p in file_list]

    with session_scope() as sess:
        envmap = sess.query(models.Envmap).get(13)
        materials = sess.query(models.Material).all()
        mat_by_id = {m.id: m for m in materials}

        app.init()

        engine = Engine.CYCLES
        scene = brender.Scene(app, shape=_REND_SHAPE,
                              engine=engine,
                              tile_size=(40, 40),
                              aa_samples=128,
                              diffuse_samples=3,
                              specular_samples=3,
                              background_mode=BackgroundMode.ENVMAP,
                              background_color=(1, 1, 1, 1))
        envmap_rotation = (0, 0, (envmap.azimuth + 3 * math.pi/2))
        scene.set_envmap(envmap.get_data_path('hdr.exr'),
                         scale=1.0, rotation=envmap_rotation)

        floor = Plane((0, 0, 0), radius=1000)

        camera = brender.BasicCamera(
            scene,
            position=(0.217, -22.76, 7.49),
            rotation=(0.99, 0, 0),
        )
        scene.set_active_camera(camera)

        grid_width = 13
        grid_height = int(math.ceil(len(inference_paths) / grid_width)) + 10
        grid_coords = []
        for r, row in enumerate(np.linspace(-grid_height/2, grid_height/2, grid_height)):
            offset = 0
            if r % 2 == 0:
                offset = 0.5
            for c, col in enumerate(np.linspace(-grid_width/2, grid_width/2, grid_width)):
                if r == 0 and c in {0, 1, grid_width - 1, grid_width - 2}:
                    continue
                if r in {1, 2, 3} and c in {0, grid_width - 1}:
                    continue

                coord = (col + offset, row)
                grid_coords.append(coord)

        random.seed(1000)
        random.shuffle(inference_paths)

        pbar = tqdm(inference_paths)
        for i, inference_path in enumerate(pbar):
            with inference_path.open('r') as f:
                inference_dict = json.load(f)

            pair = sess.query(ExemplarShapePair).get(inference_dict['pair_id'])
            pbar.set_description(f"Adding pair {pair.id}")

            location = (1.5*grid_coords[i][0], 1.5*grid_coords[i][1], 0)

            mesh = add_pair_mesh(scene, pair, inference_dict, mat_by_id)
            scene.meshes.append(mesh)
            mesh.bobj.location = location
            mesh.bobj.location[2] -= mesh.compute_min_pos()

        tqdm.write(f'Saving blend file to {args.out_path!s}')
        # bpy.ops.file.make_paths_absolute()
        if args.pack_assets:
            bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(args.out_path))


def add_pair_mesh(scene,
                  pair: ExemplarShapePair,
                  pair_dict,
                  mat_by_id):
    segment_dict = pair_dict['segments']
    rk_mesh, _ = pair.shape.load()

    with suppress_stdout():
        mesh = Mesh.from_obj(scene, pair.shape.resized_obj_path,
                             name=str(pair.id))
        mesh.resize(1.0, axis=2)
        mesh.rotate(math.pi, (0, 0, 1))
        mesh.make_normals_consistent()
        mesh.enable_smooth_shading()

    if args.type == 'inferred':
        mesh_materials = mesh.get_materials()

        bmats = []

        for seg_id, seg_name in enumerate(rk_mesh.materials):
            if str(seg_id) not in segment_dict:
                continue
            mat_id = int(segment_dict[str(seg_id)]['material'][0]['id'])
            material = mat_by_id[mat_id]
            uv_ref_scale = 2 ** (material.default_scale - 3)
            # Activate only current material.
            for bobj in mesh_materials:
                if bobj.name.startswith(seg_name):
                    tqdm.write(f'[Pair {pair.id}] Settings segment {seg_id} ({seg_name}) '
                          f'to material {material.name}, bobj_name={bobj.name}')
                    bmat = loader.material_to_brender(
                        material, bobj=bobj, uv_ref_scale=uv_ref_scale)
                    bmats.append(bmat)

        # This needs to come after the materials are initialized.
        tqdm.write('Computing UV density...')
        mesh.compute_uv_density()
    elif args.type == 'none':
        mesh.set_material(BlinnPhongMaterial(diffuse_albedo=(0.1, 0.28, 0.8),
                                             specular_albedo=(0.04, 0.04, 0.04),
                                             roughness=0.1))

    return mesh



if __name__ == '__main__':
    main()
