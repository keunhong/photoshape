from pathlib import Path

import click
import bpy

from tqdm import tqdm
import visdom
import brender
import terial
from brender import Scene, Mesh
from brender.camera import Camera
from brender.utils import suppress_stdout
from terial import models
from terial.materials.loader import material_to_brender
from terial.database import session_scope
from toolbox.images import linear_to_srgb, to_8bit

vis = visdom.Visdom(env='material-preview-rendering')


scene_path = (Path(terial.__file__).parent.parent.parent / 'resources'
              / 'BlenderMaterialPreviewScenes.blend')


@click.command()
def main():
    with session_scope() as sess:
        materials = (sess.query(models.Material)
                     .filter_by(type=models.MaterialType.MDL)
                     .order_by(models.Material.id.asc())
                     .all())

    app = brender.Brender()
    app.init(do_reset=False)
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))

    solid_scene = Scene(app, (1000, 1000),
                        bscene=bpy.data.scenes['SolidMaterialScene'],
                        aa_samples=96)
    cloth_scene = Scene(app, (1000, 1000),
                        aa_samples=96,
                        bscene=bpy.data.scenes['ClothMaterialScene'])
    camera = Camera(bpy.data.objects['Camera'])
    solid_scene.set_active_camera(camera)
    cloth_scene.set_active_camera(camera)

    pbar = tqdm(materials)
    for material in pbar:
        pbar.set_description(material.name)

        if material.data_exists('previews/bmps.png'):
            continue

        if material.substance in {'plastic', 'metal', 'wood', 'polished'}:
            scene = solid_scene
            mesh = Mesh(bpy.data.objects['SolidModel'], name='SolidModel')
        elif material.substance in {'fabric', 'leather'}:
            scene = cloth_scene
            mesh = Mesh(bpy.data.objects['ClothModel'], name='ClothModel')
        else:
            scene = solid_scene
            mesh = Mesh(bpy.data.objects['SolidModel'], name='SolidModel')
            print(f'Unknown material substance {material.substance}')

        bpy.context.screen.scene = scene.bobj

        uv_ref_scale = 2 ** (material.default_scale - 4)
        bmat = material_to_brender(material, uv_ref_scale=uv_ref_scale)
        brender.mesh.set_material(mesh.bobj, bmat)
        if bmat.has_uvs:
            mesh.compute_uv_density(base_size=12.0)

        with suppress_stdout():
            rend = scene.render_to_array(format='exr')
        rend_srgb = to_8bit(linear_to_srgb(rend))
        vis.image(rend_srgb.transpose((2, 0, 1)), win='rendering',
                  opts={'title': material.name})

        material.save_data('previews/bmps.exr', rend)
        material.save_data('previews/bmps.png', rend_srgb)


if __name__ == '__main__':
    main()
