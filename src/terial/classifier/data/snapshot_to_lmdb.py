import io

import lmdb
import argparse

import ujson as json
from pathlib import Path

import msgpack
from PIL import Image
from tqdm import tqdm

from terial import models
from terial.database import session_scope
from terial.models import SplitSet

parser = argparse.ArgumentParser()
parser.add_argument(dest='snapshot_path', type=Path)
parser.add_argument(dest='lmdb_dir', type=Path)
parser.add_argument('--saturated-thres', type=float, default=0.05)
parser.add_argument('--split-set', type=str, required=True,
                    choices=['train', 'validation'])
parser.add_argument('--save-meta', action='store_true')
args = parser.parse_args()


SHAPE = (500, 500)

def parse_example_prefix(s):
    parts = s.split('/')
    client_id = parts[0].split('=')[1]
    epoch = int(parts[1].split('=')[1])
    split_set = parts[2]
    prefix = parts[3]
    return client_id, epoch, split_set, prefix


def example_to_rend_daos(sess, client, epoch, split_set, prefix=None):
    rends = sess.query(models.Rendering).filter_by(
        client=client,
        epoch=epoch,
        split_set=SplitSet[split_set.upper()],
        # prefix=prefix,
    ).all()
    return rends


def main():
    snapshot_path = args.snapshot_path
    split_set = SplitSet[args.split_set.upper()]
    lmdb_dir = args.lmdb_dir

    with snapshot_path.open('r') as f:
        snapshot_dict = json.load(f)

    mat_id_to_label = {
        int(k): v
        for k, v in snapshot_dict['mat_id_to_label'].items()
    }

    rend_daos = []

    client_epoch_tups = set()
    for example in tqdm(snapshot_dict['examples'][split_set.name.lower()]):
        client, epoch, split_set_, prefix = parse_example_prefix(example['prefix'])
        client_epoch_tups.add((client, epoch, split_set_))

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
        for client, epoch, split_set_ in client_epoch_tups:
            rend_daos.extend(
                example_to_rend_daos(sess, client, epoch, split_set_))

    print(f"Loaded {len(rend_daos)} renderings")

    if input(f"This will create an LMDB database at {lmdb_dir!s} for the "
             f"{split_set.name} set. Continue? (y/n) ") != 'y':
        return

    lmdb_dir.mkdir(exist_ok=True, parents=True)

    if args.save_meta:
        with (lmdb_dir / 'meta.json').open('w') as f:
            json.dump({
                'mat_id_to_label': mat_id_to_label,
            }, f, indent=2)

    env = lmdb.open(str(lmdb_dir / args.split_set),
                          map_size=30000*200000)
    with env.begin(write=True) as txn:
        pbar = tqdm(rend_daos)
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
