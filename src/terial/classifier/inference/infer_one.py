import argparse
import json
from pathlib import Path

import skimage.transform
import torch
import visdom
from skimage.io import imread
from torch.nn import functional as F

import numpy as np

from terial import models
from terial.classifier.inference.utils import compute_weighted_scores_single
from terial.classifier.network import RendNet3
from terial.database import session_scope
from terial.classifier import transforms
from terial.config import SUBSTANCES

vis = visdom.Visdom(env='classifier-infer-one')


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', type=Path)
parser.add_argument(dest='image_path', type=Path)
parser.add_argument(dest='mask_path', type=Path)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()


input_size = 224
input_shape = (224, 224)


def main():
    if not args.image_path.exists():
        print(f"{args.image_path!s} does not exist")
        return

    if not args.mask_path.exists():
        print(f"{args.mask_path!s} does not exist")
        return

    if not args.checkpoint_path.exists():
        print(f"{args.checkpoint_path!s} does not exist")
        return
    checkpoint_path = args.checkpoint_path

    image = imread(args.image_path)
    if len(image.shape) > 2 and image.shape[2] > 3:
        image = image[:, :, :3]
    image = skimage.transform.resize(
        image, input_shape, anti_aliasing=True, order=3,
        mode='constant', cval=1)
    image = skimage.img_as_ubyte(image)
    mask = imread(args.mask_path)
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    mask = skimage.transform.resize(
        mask, input_shape, anti_aliasing=False, order=0,
        mode='constant', cval=0).astype(bool)

    print(image.dtype, image.shape, mask.dtype)

    with (checkpoint_path.parent / 'meta.json').open('r') as f:
        meta_dict = json.load(f)
        mat_id_to_label = meta_dict['mat_id_to_label']
        label_to_mat_id = {v: k for k, v in mat_id_to_label.items()}

    with (checkpoint_path.parent / 'model_params.json').open('r') as f:
        model_params = json.load(f)

    print(f'Loading checkpoint from {checkpoint_path!s}')
    if args.cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model = RendNet3.from_checkpoint(checkpoint)
    model.train(False)
    if args.cuda:
        model = model.cuda()

    with session_scope() as sess:
        materials = sess.query(models.Material).all()
        mat_by_id = {m.id: m for m in materials}

    vis.image(image.transpose((2, 0, 1)), win='image')
    vis.image((mask * 255).astype(np.uint8), win='mask')

    topk_dict = compute_topk(
        label_to_mat_id, model, image, mask,
        mat_by_id=mat_by_id)

    compute_weighted_scores_single(topk_dict, mat_by_id, sort=True,
                                   force_substances=False,
                                   weight_substances=True)

    k = 5
    topk_mat_ids = [p['id'] for p in topk_dict['material'][:k]]
    for i, mat_id in enumerate(topk_mat_ids):
        material = mat_by_id[mat_id]
        preview = imread(material.get_data_path('previews/bmps.png')).transpose((2, 0, 1))

        vis.images(preview, win=f'pred-{i}', opts={
            'title': f'pred-{i} (mat_id={mat_id})',
            'width': 200,
            'height': 200,
        })

    vis.text(json.dumps((topk_dict['substance']), indent=2),
             win='substance-prediction')

    print(topk_dict)


def compute_topk(label_to_mat_id,
                 model: RendNet3,
                 image,
                 seg_mask,
                 *,
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
    input_tensor = torch.cat((image_tensor, mask_tensor), dim=0).unsqueeze(0)
    if args.cuda:
        input_tensor = input_tensor.cuda()

    output = model.forward(input_tensor)

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
