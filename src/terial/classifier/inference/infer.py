import argparse
import json
from pathlib import Path

import skimage.transform
import torch
import visdom
from torch.nn import functional as F
from torchvision.models import resnet

import numpy as np
from tqdm import tqdm

from terial import models, config, controllers
from terial.classifier.network import RendNet3
from terial.database import session_scope
from toolbox.images import resize
from terial.classifier import utils, transforms
from terial.config import SUBSTANCES

vis = visdom.Visdom(env='brdf-classifier-inference')


input_size = 224


parser = argparse.ArgumentParser()
parser.add_argument(dest='checkpoint_path', type=Path)
parser.add_argument('--out-name', type=str)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()


def main():
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_name = checkpoint_path.parent.parent.name
    snapshot_path = checkpoint_path.parent.parent.parent.parent / 'snapshots' / checkpoint_name / 'snapshot.json'

    with snapshot_path.open('r') as f:
        mat_id_to_label = json.load(f)['mat_id_to_label']
        label_to_mat_id = {v: k for k, v in mat_id_to_label.items()}

    with (checkpoint_path.parent / 'model_params.json').open('r') as f:
        params_dict = json.load(f)

    print(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)

    if not args.out_name:
        # TOOD: remove this ugly thing. (There's no reason to the +1 we did)
        out_name = str(checkpoint['epoch'] - 1)
    else:
        out_name = args.out_name

    model_name = checkpoint_path.parent.name
    out_dir = (checkpoint_path.parent.parent.parent.parent
               / 'inference' / checkpoint_name / model_name / str(out_name))

    model = RendNet3(num_classes=len(label_to_mat_id) + 1,
                     num_roughness_classes=20,
                     num_substances=len(SUBSTANCES),
                     base_model=resnet.resnet18(pretrained=False),
                     output_substance=True,
                     output_roughness=True)
    model.load_state_dict(checkpoint['state_dict'])

    # model = RendNet3.from_checkpoint(checkpoint)
    model.train(False)
    model = model.cuda()

    yy = input(f'Will save to {out_dir!s}, continue? (y/n): ')
    if yy != 'y':
        return

    out_dir.mkdir(exist_ok=True, parents=True)

    print(f'Loading pairs')
    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs_default(sess)
        materials = sess.query(models.Material).all()
        mat_by_id = {m.id: m for m in materials}

    pbar = tqdm(pairs)
    for pair in pbar:
        out_path = Path(out_dir, f'{pair.id}.json')
        if not args.overwrite and out_path.exists():
            continue

        if not pair.data_exists(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME_OLD):
            continue
        pbar.set_description(f'Pair {pair.id}')

        exemplar = pair.exemplar
        shape = (224, 224)
        exemplar_im = resize(pair.exemplar.load_cropped_image(), shape)
        # if not exemplar.data_exists(exemplar.get_image_name(shape)):
        #     exemplar_im = resize(pair.exemplar.load_cropped_image(), shape)
        #     exemplar.save_data(exemplar.get_image_name(shape), exemplar_im)
        # else:
        #     exemplar_im = exemplar.load_data(exemplar.get_image_name(shape))

        segment_map = pair.load_data(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME_OLD) - 1

        vis.image(exemplar_im.transpose((2, 0, 1)), win='exemplar-image')

        result_dict = {
            'pair_id': pair.id,
            'segments': {}
        }

        for seg_id in [s for s in np.unique(segment_map) if s >= 0]:
            seg_mask = (segment_map == seg_id)
            topk_dict = compute_topk(
                label_to_mat_id, model, exemplar_im, seg_mask)
            result_dict['segments'][str(seg_id)] = topk_dict

        with open(Path(out_path), 'w') as f:
            json.dump(result_dict, f, indent=2)


def compute_topk(label_to_mat_id, model: RendNet3, image, seg_mask):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    seg_mask = skimage.transform.resize(
        seg_mask, (224, 224), order=0, anti_aliasing=False, mode='reflect')
    seg_mask = seg_mask[:, :, np.newaxis].astype(dtype=np.uint8) * 255

    image_tensor = transforms.inference_image_transform(224)(image)
    mask_tensor = transforms.inference_mask_transform(224)(seg_mask)
    input_tensor = torch.cat((image_tensor, mask_tensor), dim=0).unsqueeze(0)

    input_vis = utils.visualize_input({'image': input_tensor})
    vis.image(input_vis, win='input')

    output = model.forward(input_tensor.cuda())

    topk_mat_scores, topk_mat_labels = torch.topk(
        F.softmax(output['material'], dim=1), k=10)

    topk_dict = {
        'material': [
            {
                'score': score,
                'id': label_to_mat_id[int(label)],
            } for score, label in zip(topk_mat_scores.squeeze().tolist(),
                                      topk_mat_labels.squeeze().tolist())
        ]
    }

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
