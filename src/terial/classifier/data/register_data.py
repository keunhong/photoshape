import argparse
from pathlib import Path
import ujson as json

import tqdm

from terial import models
from terial.database import session_scope

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=Path)
args = parser.parse_args()


def main():
    if not args.dataset_path.exists():
        print(f'Given path does not exist.')
        return

    dataset_name = args.dataset_path.name

    print('Determining number of paths.')
    count = len(list(args.dataset_path.glob('**/*params.json')))

    pbar = tqdm.tqdm(total=count)

    with session_scope() as sess:
        materials = sess.query(models.Material).all()
        material_by_id = {m.id: m for m in materials}
        for client_dir in args.dataset_path.iterdir():
            client = client_dir.name.split('=')[1]
            for epoch_dir in client_dir.iterdir():
                epoch = int(epoch_dir.name.split('=')[1])
                for split_set_dir in epoch_dir.iterdir():
                    split_set = models.SplitSet[split_set_dir.name.upper()]
                    for path in split_set_dir.glob('*.params.json'):
                        pbar.update(1)
                        prefix = path.name.split('.')[0]
                        if (sess.query(models.Rendering)
                                .filter_by(dataset_name=dataset_name,
                                           client=client,
                                           split_set=split_set,
                                           epoch=epoch,
                                           prefix=prefix).count() > 0):
                            continue
                        rendering = register(
                            sess, dataset_name, client, epoch, split_set,
                            prefix, path, material_by_id)
                        pbar.set_description(f'{rendering.id}')


def register(sess, dataset, client, epoch, split_set, prefix, path,
             materials_by_id):
    with path.open('r') as f:
        params = json.load(f)

    material_ids = set(params['segment']['materials'].values())

    pair_id, index = prefix.split('_')
    rendering = models.Rendering(
        dataset_name=dataset,
        client=client,
        epoch=epoch,
        split_set=split_set,
        pair_id=int(pair_id),
        index=int(index),
        prefix=prefix,
        saturated_frac=params['saturated_frac'],
        rend_time=params['time_elapsed'],
    )

    for material_id in material_ids:
        rendering.materials.append(materials_by_id[material_id])

    sess.add(rendering)
    sess.commit()

    return rendering


if __name__ == '__main__':
    main()
