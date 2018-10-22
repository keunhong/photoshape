import argparse
import json
from pathlib import Path

import skimage.transform
import torch
import visdom
from torch.nn import functional as F

import numpy as np
from tqdm import tqdm

from terial import models, config, controllers
from terial.classifier.network import RendNet3
from terial.classifier.utils import ColorBinner
from terial.database import session_scope
from terial.classifier import utils, transforms
from terial.config import SUBSTANCES
from terial.models import ExemplarShapePair
from terial.pairs.utils import compute_segment_substances
from toolbox.images import resize

vis = visdom.Visdom(env='brdf-classifier-inference')


parser = argparse.ArgumentParser()
parser.add_argument(dest='checkpoint_path', type=Path)
parser.add_argument('--out-name', type=str)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--use-minc-substances', action='store_true')
parser.add_argument('--category', type=str, required=True)
args = parser.parse_args()


input_size = 224


def main():
    checkpoint_path = args.checkpoint_path
    base_dir = checkpoint_path.parent.parent.parent.parent
    snapshot_name = checkpoint_path.parent.parent.name
    lmdb_dir = (base_dir / 'lmdb' / snapshot_name)
    with (lmdb_dir / 'meta.json').open('r') as f:
        meta_dict = json.load(f)
        mat_id_to_label = meta_dict['mat_id_to_label']
        label_to_mat_id = {v: k for k, v in mat_id_to_label.items()}

    with (checkpoint_path.parent / 'model_params.json').open('r') as f:
        model_params = json.load(f)

    color_binner = None
    if 'color_hist_space' in model_params:
        color_binner = ColorBinner(
            space=model_params['color_hist_space'],
            shape=tuple(model_params['color_hist_shape']),
            sigma=tuple(model_params['color_hist_sigma']),
        )

    print(f'Loading checkpoint from {checkpoint_path!s}')
    checkpoint = torch.load(checkpoint_path)

    if not args.out_name:
        # TODO: remove this ugly thing. (There's no reason to the +1 we did)
        out_name = str(checkpoint['epoch'] - 1)
    else:
        out_name = args.out_name

    model_name = checkpoint_path.parent.name
    out_dir = (base_dir / 'inference' / snapshot_name / model_name / out_name)

    model = RendNet3.from_checkpoint(checkpoint)
    model.train(False)
    model = model.cuda()

    yy = input(f'Will save to {out_dir!s}, continue? (y/n): ')
    if yy != 'y':
        return

    out_dir.mkdir(exist_ok=True, parents=True)

    filters = []
    if args.category:
        filters.append(ExemplarShapePair.shape.has(category=args.category))

    print(f'Loading pairs')
    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs_default(sess, filters=filters)
        materials = sess.query(models.Material).all()
        mat_by_id = {m.id: m for m in materials}

    pairs = [
        pair for pair in pairs
        if args.overwrite or not (Path(out_dir, f'{pair.id}.json').exists())
    ]

    pbar = tqdm(pairs)
    for pair in pbar:
        out_path = Path(out_dir, f'{pair.id}.json')
        if not args.overwrite and out_path.exists():
            continue

        if not pair.data_exists(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME):
            tqdm.write(f'clean segment map not exists')
            continue
        pbar.set_description(f'Pair {pair.id}')

        exemplar = pair.exemplar
        shape = (224, 224)
        exemplar_im = pair.exemplar.load_cropped_image()
        exemplar_im = skimage.transform.resize(
            exemplar_im, shape, anti_aliasing=True, order=3,
            mode='constant', cval=1)
        # if not exemplar.data_exists(exemplar.get_image_name(shape)):
        #     exemplar_im = resize(pair.exemplar.load_cropped_image(),
        #                          shape, order=3)
        #     exemplar.save_data(exemplar.get_image_name(shape), exemplar_im)
        # else:
        #     exemplar_im = exemplar.load_data(exemplar.get_image_name(shape))

        segment_map = pair.load_data(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME) - 1
        substance_map = pair.exemplar.load_data(config.EXEMPLAR_SUBST_MAP_NAME)
        substance_map = resize(substance_map, segment_map.shape, order=0)

        vis.image(exemplar_im.transpose((2, 0, 1)), win='exemplar-image')

        result_dict = {
            'pair_id': pair.id,
            'segments': {}
        }

        subst_id_by_seg_id = compute_segment_substances(
            pair,
            return_ids=True,
            segment_map=segment_map,
            substance_map=substance_map)

        for seg_id in [s for s in np.unique(segment_map) if s >= 0]:
            seg_mask = (segment_map == seg_id)
            topk_dict = compute_topk(
                label_to_mat_id, model, exemplar_im, seg_mask,
                minc_substance=SUBSTANCES[subst_id_by_seg_id[seg_id]],
                color_binner=color_binner,
                mat_by_id=mat_by_id)
            result_dict['segments'][str(seg_id)] = topk_dict

        with open(Path(out_path), 'w') as f:
            json.dump(result_dict, f, indent=2)


def compute_topk(label_to_mat_id,
                 model: RendNet3,
                 image,
                 seg_mask,
                 *,
                 color_binner,
                 minc_substance,
                 mat_by_id):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)


    seg_mask = skimage.transform.resize(
        seg_mask, (224, 224), order=0, anti_aliasing=False, mode='constant')
    seg_mask = seg_mask[:, :, np.newaxis].astype(dtype=np.uint8) * 255

    image_tensor = transforms.inference_image_transform(
        input_size=224, output_size=224, pad=0, to_pil=True)(image)
    mask_tensor = transforms.inference_mask_transform(
        input_size=224, output_size=224, pad=0)(seg_mask)
    input_tensor = (torch.cat((image_tensor, mask_tensor), dim=0)
                    .unsqueeze(0).cuda())

    input_vis = utils.visualize_input({'image': input_tensor})
    vis.image(input_vis, win='input')

    output = model.forward(input_tensor)

    if 'color' in output:
        color_output = output['color']
        color_hist_vis = color_binner.visualize(
            F.softmax(color_output[0].cpu().detach(), dim=0))
        vis.heatmap(color_hist_vis, win='color-hist',
                    opts=dict(title='Color Histogram'))

    topk_mat_scores, topk_mat_labels = torch.topk(
        F.softmax(output['material'], dim=1), k=output['material'].size(1))

    topk_dict = {'material': list()}

    for score, label in zip(topk_mat_scores.squeeze().tolist(),
                            topk_mat_labels.squeeze().tolist()):
        if int(label) == 0:
            continue
        mat_id = int(label_to_mat_id[int(label)])
        material = mat_by_id[mat_id]
        topk_dict['material'].append({
            'score': score,
            'id': mat_id,
            'pred_substance': material.substance,
            'minc_substance': minc_substance,
        })

    if 'substance' in output:
        topk_subst_scores, topk_subst_labels = torch.topk(
            F.softmax(output['substance'].cpu(), dim=1),
            k=output['substance'].size(1))
        topk_dict['substance'] = \
            [
                {
                    'score': score,
                    'id': label,
                    'name': SUBSTANCES[int(label)],
                } for score, label in zip(topk_subst_scores.squeeze().tolist(),
                                          topk_subst_labels.squeeze().tolist())
            ]

    if 'roughness' in output:
        nrc = model.num_roughness_classes
        roughness_midpoints = np.linspace(1/nrc/2, 1-1/nrc/2, nrc)
        topk_roughness_scores, topk_roughness_labels = torch.topk(
            F.softmax(output['roughness'].cpu(), dim=1), k=5)
        topk_dict['roughness'] = \
            [
                {
                    'score': score,
                    'value': roughness_midpoints[int(label)],
                } for score, label in zip(topk_roughness_scores.squeeze().tolist(),
                                          topk_roughness_labels.squeeze().tolist())
            ]

    return topk_dict


if __name__ == '__main__':
    main()
