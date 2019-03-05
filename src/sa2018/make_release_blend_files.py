import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List

from tqdm import tqdm

from brender.utils import suppress_stdout
from terial import controllers, models
from terial.database import session_scope




parser = argparse.ArgumentParser()
parser.add_argument('--shape-source', type=str)
parser.add_argument('--shape-category', type=str)
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--max-dist', type=float, default=16.0)
parser.add_argument('--out-dir', type=Path, required=True)
parser.add_argument('--type', default='inferred',
                    choices=['inferred', 'mtl'])
args = parser.parse_args()


def main():
    args.out_dir.mkdir(exist_ok=True, parents=True)

    filters = []
    if args.shape_source:
        filters.append(models.Shape.source == args.shape_source)

    if args.shape_category:
        filters.append(models.Shape.category == args.shape_category)

    if args.topk:
        filters.append(models.ExemplarShapePair.rank <= args.topk)

    inference_dir = "/projects/grail/kparnb/photoshape/brdf-classifier" \
                    "/inference/20180516-500x500/20180527.022554.resnet34" \
                    ".opensurface_pretrained_on_substance_only.subst_loss=fc" \
                    ".color_loss=none.lab_10,10,10_1.0,1.0," \
                    "1.0.lr=0.0001.mask_noise_p=0.0.use_variance.sanka/45"

    with session_scope() as sess:
        shapes, count, pair_count = controllers.fetch_shapes_with_pairs(
            sess, filters=filters, max_dist=args.max_dist)
        print(f"Fetched {count} shapes.")
        for shape in tqdm(shapes):
            pairs: List[models.ExemplarShapePair] = shape.get_topk_pairs(args.topk, args.max_dist)
            for pair_rank, pair in enumerate(tqdm(pairs)):
                json_path = Path(inference_dir, f'{pair.id}.json')
                if json_path.exists():
                    make_blend_file(json_path, args.out_dir, shape, pair, pair_rank)


def make_blend_file(inference_path, out_dir, shape, pair, pair_rank):

    prefix = f'shape{shape.id:05d}_rank{pair_rank:02d}_pair{pair.id}'
    blend_name = f'{prefix}.blend'
    blend_out_path = out_dir / blend_name

    ext = pair.exemplar.cropped_path.suffix
    exemplar_name = f'{prefix}.exemplar{ext}'
    exemplar_out_path = out_dir / exemplar_name
    shutil.copy(pair.exemplar.cropped_path, exemplar_out_path)

    command = [
        'python', '-m', 'terial.classifier.rendering.create_blend_file',
        str(inference_path), str(blend_out_path),
        '--pack-assets',
        '--type', args.type,
        '--use-weighted-scores',
    ]
    print(command)

    command_str = ' '.join(command)
    tqdm.write(f'Launching command {command_str!r}')

    subprocess.call(command)

    tqdm.write('Done!')


if __name__ == '__main__':
    main()
