from collections import Counter

import numpy as np
from scipy.misc import imresize

from kitnn.models import minc
from terial import config
from terial.config import SUBSTANCES
from terial.models import ExemplarShapePair
from toolbox.images import resize, mask_bbox, bbox_shape, bbox_make_square


def pair_uncropped_exemplar(pair: ExemplarShapePair):
    fg_mask = pair.load_data(config.PAIR_FG_BBOX_NAME)
    return compute_uncropped_exemplar(pair.exemplar.load_cropped_image(), fg_mask)



def compute_uncropped_exemplar(exemplar_im, fg_mask):
    fg_bbox = mask_bbox(fg_mask)
    fg_bbox = bbox_make_square(fg_bbox)
    in_shape = bbox_shape(fg_bbox)
    print(in_shape)
    in_image = imresize(exemplar_im, in_shape)
    out_image = np.full((*fg_mask.shape, 3), dtype=np.uint8, fill_value=255)
    out_image[fg_bbox[0]:fg_bbox[1], fg_bbox[2]:fg_bbox[3]] = in_image
    return out_image



def compute_substance_ids_by_segment(subst_map, segment_map):
    seg_subst_ids = {}
    seg_ids = [i for i in np.unique(segment_map) if i >= 0]
    for seg_id in np.unique(seg_ids):
        most_common = Counter(subst_map[segment_map == seg_id]).most_common(6)
        most_common = [(k, v) for k, v in most_common if k >= 0]
        seg_subst_ids[seg_id] = [
            i for i, _ in most_common if i < len(minc.REMAPPED_SUBSTANCES) - 1]
        if len(seg_subst_ids[seg_id]) == 0:
            seg_subst_ids[seg_id] = -1
        else:
            seg_subst_ids[seg_id] = seg_subst_ids[seg_id][0]
    return seg_subst_ids


def compute_segment_substance_map(segment_map, seg_subst_ids):
    seg_subst_masks = {}
    seg_ids = [i for i in np.unique(segment_map) if i >= 0]
    for seg_id in np.unique(seg_ids):
        seg_subst_masks[seg_id] = (segment_map == seg_id)

    seg_subst_map = np.full(segment_map.shape, dtype=int,
                            fill_value=SUBSTANCES.index('background'))
    for seg_id, subst_id in seg_subst_ids.items():
        mask = seg_subst_masks[seg_id]
        seg_subst_map[mask] = subst_id

    return seg_subst_map


def compute_segment_substances(pair: ExemplarShapePair,
                               return_ids=False,
                               segment_map=None,
                               substance_map=None):
    if segment_map is None:
        segment_map = pair.load_data(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME) - 1
    if substance_map is None:
        substance_map = pair.exemplar.load_data(config.EXEMPLAR_SUBST_MAP_NAME)
        substance_map = resize(substance_map, segment_map.shape, order=0)

    mesh, _ = pair.shape.load()
    seg_substances = compute_substance_ids_by_segment(substance_map,
                                                      segment_map)
    if return_ids:
        seg_substances = {
            k: v for k, v in seg_substances.items()
        }
    else:
        seg_substances = {
            mesh.materials[k]: minc.REMAPPED_SUBSTANCES[v]
            for k, v in seg_substances.items()
        }
    return seg_substances
