import argparse
import bpy

import os

from terial.shapes.shapenet import get_synset_models, taxonomy
from toolbox.logging import init_logger

logger = init_logger(__name__)

MAX_SIZE = 5e7


_package_dir = os.path.dirname(os.path.realpath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, required=True)
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()


def reset_blender():
    bpy.ops.wm.read_factory_settings()

    for scene in bpy.data.scenes:
        for obj in scene.objects:
            scene.objects.unlink(obj)

    # only worry about data in the startup scene
    for bpy_data_iter in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.lamps,
            bpy.data.cameras,
    ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data)


def main():
    logger.info("Loading models.")
    models = get_synset_models(taxonomy.name_to_id(args.category))
    for model_idx, model in enumerate(models):
        if model_idx < args.start:
            continue
        model_pfx = "[model {}/{}]".format(model_idx + 1, len(models))
        logger.info("{} Processing {}".format(model_pfx, model.orig_obj_path))
        out_path = os.path.join(model.path, 'models/model_processed_v2.obj')
        # if model.model_id in config.MODEL_BLACKLIST:
        #     logger.info("Skipping blacklisted model.")
        #     continue
        if os.path.exists(out_path):
            logger.info("Already exists.")
            continue
        model_size = os.path.getsize(model.orig_obj_path)
        if model_size > MAX_SIZE:
            logger.warning("Model too big ({} > {})"
                           .format(model_size, MAX_SIZE))
            continue
        logger.info("Resetting blender.")
        reset_blender()
        logger.info("Importing OBJ to blender.")
        bpy.ops.import_scene.obj(filepath=model.orig_obj_path,
                                 use_edges=True, use_smooth_groups=True,
                                 use_split_objects=True, use_split_groups=True,
                                 use_groups_as_vgroups=False,
                                 use_image_search=True)

        # if len(bpy.data.objects) > 100:
        #     logger.info("Too many objects. Skipping for now..")
        #     continue

        for obj_idx, obj in enumerate(bpy.data.objects):
            logger.info("{}[obj {}/{}] Processing object.".format(
                model_pfx, obj_idx + 1, len(bpy.data.objects)))
            bpy.context.scene.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            logger.info("Clearing split normals and removing doubles.")
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
            bpy.ops.mesh.remove_doubles()

            bpy.ops.mesh.select_all(action='DESELECT')
            obj.select = True

            logger.info("Unchecking auto_smooth")
            obj.data.use_auto_smooth = False

            bpy.ops.object.modifier_add(type='EDGE_SPLIT')
            logger.info("Adding edge split modifier.")
            mod = obj.modifiers['EdgeSplit']
            mod.split_angle = 20

            bpy.ops.object.mode_set(mode='OBJECT')

            logger.info("Applying smooth shading.")
            bpy.ops.object.shade_smooth()

            logger.info("Running smart UV project.")
            bpy.ops.uv.smart_project()

            obj.select = False
        bpy.ops.export_scene.obj(filepath=out_path,
                                 group_by_material=True,
                                 keep_vertex_order=True,
                                 use_normals=True, use_uvs=True,
                                 use_materials=True)
        logger.info("Saved to {}".format(out_path))


if __name__ == '__main__':
    reset_blender()
    main()
