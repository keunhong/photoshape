import io

import lmdb
import argparse

import ujson as json
from pathlib import Path

import msgpack
from PIL import Image
from tqdm import tqdm
import sqlalchemy as sa

from terial import models
from terial.database import session_scope
from terial.models import SplitSet

parser = argparse.ArgumentParser()
parser.add_argument(dest='dataset', type=str)
parser.add_argument(dest='snapshot_dir', type=Path)
parser.add_argument('--saturated-thres', type=float, default=0.05)
parser.add_argument('--split-set', type=str, required=True)
parser.add_argument('--save-meta', action='store_true')
args = parser.parse_args()


SHAPE = (500, 500)


def main():
    snapshot_dir = args.snapshot_dir

    with session_scope() as sess:
        enabled_materials = (sess.query(models.Material)
                      .filter_by(enabled=True)
                      .order_by(models.Material.substance,
                                models.Material.type,
                                models.Material.source,
                                models.Material.name)
                      .all())
        material_by_id = {
            m.id: m
            for m in enabled_materials
        }

    print(f"Fetched {len(enabled_materials)} enabled materials.")

    split_set = SplitSet[args.split_set.upper()]

    if input(f"This will create an LMDB database at {snapshot_dir!s} for the "
             f"{split_set.name} set. Continue? (y/n) ") != 'y':
        return

    snapshot_dir.mkdir(exist_ok=True, parents=True)

    mat_id_to_label = {
        # Reserve 0 for background.
        material.id: i + 1 for i, material in enumerate(enabled_materials)
    }
    mat_id_to_base_color = {
        material.id: material.params['base_color_hist']
        for material in enabled_materials
    }

    if args.save_meta:
        with (snapshot_dir / 'meta.json').open('w') as f:
            json.dump({
                'dataset': args.dataset,
                'mat_id_to_label': mat_id_to_label,
                'mat_id_to_base_color_hist': mat_id_to_base_color,
            }, f, indent=2)

    tqdm.write(f"Fetching renderings from DB.")
    with session_scope() as sess:
        rends = (
            sess.query(models.Rendering)
                .filter(sa.and_(
                    models.Rendering.dataset_name == args.dataset,
                    models.Rendering.split_set == split_set,
                    models.Rendering.saturated_frac < args.saturated_thres,
                    sa.not_(models.Rendering.exclude)))
                .all())

    train_env = lmdb.open(str(snapshot_dir / args.split_set),
                          map_size=30000*200000)
    with train_env.begin(write=True) as txn:
        pbar = tqdm(rends)
        rend: models.Rendering
        for rend in pbar:
            pbar.set_description(f'{rend.id}')
            ldr_im = Image.open(rend.get_ldr_path(SHAPE))
            seg_map = Image.open(rend.get_segment_map_path(SHAPE))
            ldr_bytes = io.BytesIO()
            ldr_im.save(ldr_bytes, format='JPEG')
            seg_map_bytes = io.BytesIO()
            seg_map.save(seg_map_bytes, format='PNG')

            rend_params = rend.load_params()
            mat_id_by_seg_name = rend_params['segment']['materials']

            valid = True
            for mat_id in mat_id_by_seg_name.values():
                if int(mat_id) not in material_by_id:
                    valid = False
                    break
                material = material_by_id[int(mat_id)]
                if not material.enabled:
                    valid = False
                    break
                if mat_id not in mat_id_to_label:
                    valid = False
                    break
            if not valid:
                continue

            seg_material_ids = {
                seg_id: mat_id_by_seg_name[seg_name]
                for seg_name, seg_id in rend_params['segment']['segment_ids'].items()
                if seg_name in mat_id_by_seg_name
            }
            seg_substances = {
                seg_id: material_by_id[mat_id].substance
                for seg_id, mat_id in seg_material_ids.items()
            }

            if len(seg_substances) == 0:
                continue

            payload = msgpack.packb({
                'rend_id': rend.pair_id,
                'pair_id': rend.id,
                'ldr_image': ldr_bytes.getvalue(),
                'segment_map': seg_map_bytes.getvalue(),
                'seg_material_ids': seg_material_ids,
                'seg_substances': seg_substances,
            })
            txn.put(f'{rend.id:08d}'.encode(), payload)


if __name__ == '__main__':
    main()
