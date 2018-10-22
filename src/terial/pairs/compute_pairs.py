"""
gF
Generates exemplar-shape pairs.
"""
from pathlib import Path

import click as click
import collections
import numpy as np
import sqlalchemy as sa
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from terial import config
from terial.database import session_scope
from terial.models import Shape, Exemplar, ExemplarShapePair


@click.command()
@click.option('--batch-size', default=400)
@click.option('--num-workers', default=1)
@click.option('--start', default=0)
@click.option('--end', type=int)
@click.option('--topk', type=int, default=8)
@click.option('--prefetch', is_flag=True)
@click.option('--category', type=str)
def main(start, end, batch_size, num_workers, topk, prefetch, category):
    with session_scope() as sess:
        tqdm.write('Fetching shapes.')
        shapes = (sess.query(Shape)
                  .filter_by(exclude=False, category=category)
                  .order_by(Shape.id.asc())
                  .all())
        tqdm.write('Fetching exemplars.')
        filters = [Exemplar.id >= start,
                   Exemplar.exclude == False,
                   Exemplar.category == category]
        if end:
            filters.append(Exemplar.id < end)

        exemplars = (sess.query(Exemplar)
                     .filter(sa.and_(*filters))
                     .order_by(Exemplar.id.asc())
                     .all())

    tqdm.write('Filtering shapes and exemplars with aligning features.')
    valid_shapes = [shape for shape in shapes
                    if shape.data_exists(config.SHAPE_ALIGN_DATA_NAME)]
    valid_exemplars = [exemplar for exemplar in exemplars
                       if exemplar.data_exists(config.EXEMPLAR_ALIGN_DATA_NAME)]
    tqdm.write(f'Using {len(valid_shapes)}/{len(shapes)} shapes, '
               f'{len(valid_exemplars)}/{len(exemplars)} exemplars')

    dataset = AlignFeatureDataset(valid_shapes, prefetch=prefetch)
    shape_loader = DataLoader(
        dataset=dataset, sampler=SequentialSampler(dataset),
        shuffle=False, batch_size=batch_size, num_workers=num_workers)

    pair_matches_by_exemplar = collections.defaultdict(list)

    # Iterate over shapes in outer loop since loading them is considerably more
    # expensive.
    with session_scope() as sess:
        exemplar_pbar = tqdm(valid_exemplars)
        for exemplar in exemplar_pbar:
            n_existing = sess.query(
                ExemplarShapePair).filter_by(exemplar_id=exemplar.id).count()

            if n_existing >= topk:
                exemplar_pbar.set_description(
                    f'Exemplar {exemplar.id} already has pairs, skipping')
                continue
            exemplar_pbar.set_description(f'Exemplar {exemplar.id}')
            process_exemplar(sess, exemplar, shape_loader, topk=topk)

    pbar = tqdm(pair_matches_by_exemplar.items())
    pbar.set_description('Choosing top-k matches')

    n_pairs = 0

    with session_scope(commit=True) as sess:
        for exemplar_id, exemplar_pairs in pbar:
            exemplar_pairs = [pair for pair in exemplar_pairs if pair.distance
                              < config.ALIGN_DIST_THRES_GEN]
            exemplar_pairs.sort(key=lambda pair: pair.distance)
            for pair in exemplar_pairs[:config.ALIGN_TOP_K]:
                n_pairs += 1
                sess.add(pair)

    tqdm.write(f'Saved {n_pairs} pairs')


def normalize_feats(batch):
    feat_norms = batch.norm(2, dim=1).view(batch.size(0), 1).expand(
        *batch.size())
    feats = (batch / feat_norms).cpu()
    return feats


def compute_feat_dists(batch_feats, image_feats):
    batch_feats = normalize_feats(batch_feats)
    image_feats = normalize_feats(image_feats.view(1, -1))
    return (batch_feats.cuda() - image_feats.cuda().expand(
        *batch_feats.size())).pow(2).sum(dim=1)


def process_exemplar(sess, exemplar, shape_loader, topk, max_dist=40):
    shape_pbar = tqdm(shape_loader)

    best_pairs = []

    for batch_idx, batch in enumerate(shape_pbar):
        shape_ids, shape_feats, thetas, phis, fovs = batch

        shape_feats = shape_feats.float().cuda()
        shape_pbar.set_description(f'Batch {batch_idx}')

        exemplar_feat = exemplar.load_data(config.EXEMPLAR_ALIGN_DATA_NAME)
        exemplar_feat = (torch.from_numpy(exemplar_feat)
                         # Dimensions: (shape, viewpoint, feat_dims).
                         .view(1, 1, -1)
                         .float()
                         .cuda())

        # L2 dist from exemplar feature for all shapes/viewpoints.
        # Output shape: (batch_size, 456)
        dists = ((shape_feats - exemplar_feat.expand(*shape_feats.size()))
                 .pow(2).sum(dim=2))
        dists, vp_inds = torch.min(dists.cpu(), dim=1)
        batch_best_dist, batch_best_ind = torch.min(dists, dim=0)

        i = int(batch_best_ind.cpu().item())
        dist = float(batch_best_dist.cpu().item())
        if dist > max_dist:
            continue

        vp_ind = vp_inds[batch_best_ind]
        # If we haven't collect at least top-k pairs or this is better than the
        # worst collected one.
        if len(best_pairs) < topk or dist < best_pairs[-1].distance:
            pair = ExemplarShapePair(
                exemplar_id=exemplar.id,
                shape_id=shape_ids[i].item(),
                azimuth=float(thetas[i][vp_ind]),
                elevation=float(phis[i][vp_ind]),
                fov=float(fovs[i][vp_ind]),
                distance=dist,
                feature_type=config.SHAPE_ALIGN_DATA_NAME)
            # Add pair to list and remove previous match.
            best_pairs.append(pair)
            best_pairs.sort(key=lambda p: p.distance)
            best_pairs = best_pairs[:topk]

    num_added = 0
    for pair in best_pairs:
        existing_count = sess.query(ExemplarShapePair).filter_by(
            shape_id=pair.shape_id,
            exemplar_id=pair.exemplar_id,
        ).count()

        if existing_count > 0:
            continue
        else:
            sess.add(pair)
            num_added += 1

    sess.commit()

    tqdm.write(f"Exemplar {exemplar.id}: "
               f"Added {num_added} pairs "
               f"({len(best_pairs) - num_added} were duplicates)")


class AlignFeatureDataset(Dataset):
    def __init__(self, shapes, prefetch=False):
        self.shapes = shapes

        self.data_list = []
        self.prefetch = prefetch

        if prefetch:
            tqdm.write('Prefetching shape data')
            path = Path('/local1/kpar/data/terial/shape_data.pth')
            if path.exists():
                tqdm.write(f"Loading shape data from {path!s}")
                self.data_list = torch.load(str(path))
            else:
                tqdm.write('Loading shape data individually')
                pbar = tqdm(shapes)
                for shape in pbar:
                    pbar.set_description(f'Shape {shape.id}')
                    self.data_list.append(self._load_data(shape))
                tqdm.write(f"Saving shape data to {path!s}")
                torch.save(self.data_list, str(path))

    @staticmethod
    def _load_data(shape):
        data = np.load(
            shape.get_data_path(config.SHAPE_ALIGN_DATA_NAME))

        # Support new format.
        if 'arr_0' in data:
            data = data['arr_0'][()]

        feats = torch.from_numpy(data['feats'][()].astype(np.float16))
        thetas = torch.from_numpy(data['thetas'][()].astype(np.float16))
        phis = torch.from_numpy(data['phis'][()].astype(np.float16))
        fovs = torch.from_numpy(data['fovs'][()].astype(np.float16))
        return shape.id, feats, thetas, phis, fovs

    def __getitem__(self, index):
        """
        The shape of the feats returned is (batch_size, n_viewpoints, feat_dims)
        which means the batch size is the number of shapes. Each shape has 456
        different viewpoints.

        :param index:
        :return:
        """

        if self.prefetch:
            shape_id, feats, thetas, phis, fovs = self.data_list[index]
        else:
            shape = self.shapes[index]
            shape_id, feats, thetas, phis, fovs = self._load_data(shape)

        return (shape_id, feats.float(), thetas.float(), phis.float(),
                fovs.float())

    def __len__(self):
        return len(self.shapes)


if __name__ == '__main__':
    main()
