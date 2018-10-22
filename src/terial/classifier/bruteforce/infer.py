from pathlib import Path

import click
import logging

import json
import skimage
import torch
import visdom
from scipy import linalg
from skimage import morphology, transform
from skimage.color import rgb2lab
from skimage.io import imread

import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from kitnn.models import minc
from kitnn.utils import make_batch
from terial import config, controllers
from terial.database import session_scope
from terial.models import ExemplarShapePair
from terial.pairs import utils
from toolbox import sampling
from toolbox.images import crop_tight_fg, mask_bbox

vis = visdom.Visdom()


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


_BATCH_SIZE = 1


# MAT_CAND_NAME = config.MATERIAL_CAND_VGG16_WHOLE_NAME
MAT_CAND_NAME = config.MATERIAL_CAND_MEDCOL_NAME
# MAT_CAND_NAME = config.MATERIAL_CAND_VGG16_PATCH_NAME

if 'minc' in MAT_CAND_NAME:
    logger.info('Loading VGG.')
    mincnet = minc.MincVGG()
    mincnet.load_npy(config.MINC_VGG16_WEIGHTS_PATH)
    mincnet = mincnet.cuda()


@click.command()
@click.argument('out_dir', type=click.Path())
def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs(
            sess,
            by_shape=True,
            # max_dist=config.ALIGN_DIST_THRES,
            order_by=ExemplarShapePair.distance.asc())

        logger.info('Fetched %d pairs. align_dist_thres = %f',
                    len(pairs), config.ALIGN_DIST_THRES)

        pair_pbar = tqdm(pairs)
        for pair in pair_pbar:
            pair_pbar.set_description(f'Pair {pair.id}')

            if not pair.data_exists(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME):
                continue

            seg_mat_candidates = assign_materials(pair)
            if seg_mat_candidates and len(seg_mat_candidates) > 0:
                with Path(out_dir, f'{pair.id}.json').open('w') as f:
                    json.dump(seg_mat_candidates, f, indent=2)


def parse_mat_path(path):
    mat_id, mat_substance, mat_type = path.name.split('.')[0].split(',')
    return mat_id, mat_substance, mat_type


def crop_largest_component(image, mask, shape):
    image = image.copy()[:, :, :3]
    image[~mask] = 1.0
    labels = morphology.label(mask)
    mask = labels == np.argmax(np.bincount(labels[labels > 0].flat))
    bbox = mask_bbox(mask)
    if MAT_CAND_NAME == config.MATERIAL_CAND_VGG16_WHOLE_NAME:
        image = crop_tight_fg(image, shape=shape,
                              bbox=bbox, fill=1.0, use_pil=True)
    else:
        image = crop_tight_fg(image, shape=shape,
                              bbox=bbox, fill=1.0, use_pil=True)
    return image[:, :, :3]


class VGG16WholeDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __getitem__(self, index):
        image = skimage.img_as_float32(imread(self.paths[index]))
        image = crop_largest_component(image, image[:, :, 3] > 0,
                                       shape=(227, 227))
        image = torch.from_numpy(
            minc.preprocess_image(image).transpose((2, 0, 1)))
        return image

    def __len__(self):
        return len(self.paths)


class MedianColorDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __getitem__(self, index):
        image = skimage.img_as_float32(imread(self.paths[index]))
        image = crop_largest_component(image, image[:, :, 3] > 0,
                                       shape=config.SHAPE_REND_SHAPE)
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        return image

    def __len__(self):
        return len(self.paths)


class VGG16PatchDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __getitem__(self, index):
        image = skimage.img_as_float32(imread(self.paths[index]))
        mask = image[:, :, 3] > 0
        return make_batch(self.rand_patches(minc.preprocess_image(image), mask))

    @staticmethod
    def rand_patches(image, mask, n_patches=7):
        bboxes, _ = sampling.generate_random_bboxes(
            mask, num_patches=n_patches, scale_range=(0.2, 0.5))
        patches = sampling.bboxes_to_patches(
            image[:, :, :3], bboxes, patch_size=227)
        vis.image(np.hstack(patches).transpose((2, 0, 1)))
        return patches

    def __len__(self):
        return len(self.paths)


def compute_features(input, feat_layer='relu7'):
    _, responses = mincnet.forward(input, selection=[feat_layer])
    features = responses[feat_layer]
    features = features.view(features.size(0), -1)
    features = features / features.norm(dim=1).unsqueeze(1)
    return features


def assign_materials(pair: ExemplarShapePair):
    exemplar_seg_map = pair.load_data(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME) - 1
    shape_seg_map = pair.load_data(config.SHAPE_REND_SEGMENT_MAP_NAME) - 1
    mesh, _ = pair.shape.load()

    exemplar_im = transform.resize(
        pair.exemplar.load_cropped_image(), config.SHAPE_REND_SHAPE,
        preserve_range=True, anti_aliasing=True, mode='constant')
    substance_map = pair.exemplar.load_data(config.EXEMPLAR_SUBST_MAP_NAME)
    seg_substances = utils.compute_segment_substances(pair)

    seg_mat_candidates = {}

    seg_pbar = tqdm(mesh.materials)
    for seg_id, seg_name in enumerate(seg_pbar):
        seg_pbar.set_description(seg_name)
        if seg_name not in seg_substances:
            logger.warning('Segment %r not in seg_substances',
                           seg_name)
            continue
        seg_substance_id = minc.REMAPPED_SUBSTANCES.index(seg_substances[seg_name])
        substance_mask = substance_map == seg_substance_id
        exemplar_seg_mask = (exemplar_seg_map == seg_id)
        shape_seg_mask = (shape_seg_map == seg_id)
        sample_mask = exemplar_seg_mask & substance_mask

        if np.sum(sample_mask) == 0:
            logger.warning('Sample mask not big enough')
            continue

        try:
            mat_exemplar_im = crop_largest_component(exemplar_im, sample_mask,
                                                     shape=(227, 227))
        except ValueError:
            logger.warning('Could not get largest component for '
                           'segment %r', seg_name)
            continue

        # pair.save_data('{}/{}'.format(config.MATERIAL_CAND_EXEMPLAR_NAME,
        #                               seg_name), mat_exemplar_im)

        mat_rend_dir = Path(
            pair.get_image_dir_path(), 'material_rends_256x256', seg_name)
        if not mat_rend_dir.exists():
            logger.warning('%r does not exist', str(mat_rend_dir))
            continue
        mat_rend_paths = list(mat_rend_dir.iterdir())
        if len(mat_rend_paths) == 0:
            logger.error('There are no rendering for segment %s', seg_name)
            continue

        if MAT_CAND_NAME == config.MATERIAL_CAND_VGG16_WHOLE_NAME:
            dataset = VGG16WholeDataset(mat_rend_paths)
            exemplar_feature = compute_features(
                Variable(torch.from_numpy(
                    minc.preprocess_image(mat_exemplar_im).astype(np.float32)
                        .transpose((2, 0, 1)))
                         .unsqueeze(0)
                         .cuda())).view(1, -1)
        elif MAT_CAND_NAME == config.MATERIAL_CAND_VGG16_PATCH_NAME:
            dataset = VGG16PatchDataset(mat_rend_paths)
            exemplar_patches = VGG16PatchDataset.rand_patches(
                minc.preprocess_image(exemplar_im), sample_mask)
            exemplar_features = compute_features(
                Variable(make_batch(exemplar_patches).cuda()))
            exemplar_features = exemplar_features.view(
                exemplar_features.size(0), -1)
        elif MAT_CAND_NAME == config.MATERIAL_CAND_MEDCOL_NAME:
            dataset = MedianColorDataset(mat_rend_paths)
            exemplar_mean_col = np.median(
                rgb2lab(exemplar_im)[sample_mask & substance_mask], axis=0)

        loader = DataLoader(dataset=dataset,
                            sampler=SequentialSampler(dataset),
                            shuffle=False,
                            batch_size=_BATCH_SIZE,
                            num_workers=8)
        pbar = tqdm(loader)
        mat_candidates = []
        for batch_idx, batch in enumerate(pbar):
            pbar.set_description('Computing distances')
            if MAT_CAND_NAME == config.MATERIAL_CAND_VGG16_WHOLE_NAME:
                features = compute_features(Variable(batch.cuda()))
                dists = [d.cpu().data[0] for d in
                         (1 - exemplar_feature @ features.t()).squeeze()]
            elif MAT_CAND_NAME == config.MATERIAL_CAND_MEDCOL_NAME:
                dists = []
                for tensor in batch:
                    rend = tensor.numpy().transpose((1, 2, 0)).clip(0, 1)
                    rend_mean_col = np.median(rgb2lab(rend)[shape_seg_mask],
                                              axis=0)
                    dists.append(linalg.norm(rend_mean_col - exemplar_mean_col))
            elif MAT_CAND_NAME == config.MATERIAL_CAND_VGG16_PATCH_NAME:
                dists = []
                for patchbatch in batch:
                    features = compute_features(
                        Variable(patchbatch.cuda()))
                    dists.append(
                        (exemplar_features @ features.t()).cpu().data.view(-1).min())

            pbar.set_description('Creating candidates')
            for i, dist in enumerate(dists):
                mat_rend_idx = batch_idx * _BATCH_SIZE + i
                mat_id, mat_subst, mat_type = parse_mat_path(
                    mat_rend_paths[mat_rend_idx])

                mat_candidates.append({
                    'id': int(mat_id),
                    'type': mat_type,
                    'substance': mat_subst,
                    'distance': dist,
                })
        mat_candidates.sort(key=lambda c: c['distance'])
        seg_mat_candidates[seg_name] = mat_candidates

    return seg_mat_candidates

    # logger.info('Saving segment material candidates.')
    # pair.save_data(MAT_CAND_NAME, seg_mat_candidates)


if __name__ == '__main__':
    main()
