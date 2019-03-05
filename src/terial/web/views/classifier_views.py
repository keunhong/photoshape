import logging
import math
import subprocess
import json
from collections import defaultdict
from pathlib import Path

import aiohttp_jinja2

import uuid
from aiohttp import web

import numpy as np
from multidict import MultiDict

import toolbox.web
from terial import config, models, controllers
from terial.classifier.inference.utils import compute_weighted_scores
from terial.database import session_scope

logger = logging.getLogger(__name__)


def get_model_inference_dir(snapshot_name, model_name):
    snapshot_dir = config.BRDF_CLASSIFIER_SNAPSHOTS_DIR / snapshot_name
    return snapshot_dir / 'inference' / model_name


@aiohttp_jinja2.template('classifier/list_models.html')
async def list_models(request: web.Request):
    models_by_snapshot = defaultdict(list)

    snapshot_paths = list(reversed(sorted(
        (config.BRDF_CLASSIFIER_DIR_REMOTE / 'lmdb').iterdir())))

    snapshot_paths.extend(reversed(sorted(
        config.BRDF_CLASSIFIER_SNAPSHOTS_DIR.iterdir())))


    for snapshot_path in snapshot_paths:
        snapshot_name = snapshot_path.name
        checkpoint_dir = Path(config.BRDF_CLASSIFIER_DIR_REMOTE,
                              'checkpoints', snapshot_name)
        if not checkpoint_dir.exists():
            continue
        model_json_paths = reversed(sorted(
            checkpoint_dir.glob('*/model_params.json')))
        for model_json_path in model_json_paths:
            model_name = model_json_path.parent.name
            with model_json_path.open('r') as f:
                model_params = json.load(f)
            best_stats_path = model_json_path.parent / 'model_best_stats.json'
            if best_stats_path.exists():
                with best_stats_path.open('r') as f:
                    best_stats = json.load(f)
            else:
                best_stats = None

            snapshot_name = snapshot_path.name

            model_inference_dir = (config.BRDF_CLASSIFIER_DIR_REMOTE / 'inference'
                                   / snapshot_name / model_name)
            if model_inference_dir.exists():
                inference_epochs = [p.name for p in model_inference_dir.iterdir()]
            else:
                inference_epochs = []

            models_by_snapshot[snapshot_name].append({
                'params': model_params,
                'best_stats': best_stats,
                'snapshot_name': snapshot_path.name,
                'model_name': model_json_path.parent.name,
                'model_path': str(model_json_path),
                'inference_epochs': inference_epochs,
            })

    return {
        'models_by_snapshot': models_by_snapshot,
    }


@aiohttp_jinja2.template('classifier/list_models_ablation.html')
async def list_models_ablation(request: web.Request):
    models_by_snapshot = defaultdict(list)

    snapshot_paths = list(reversed(sorted(
        (config.BRDF_CLASSIFIER_DIR_REMOTE / 'lmdb').iterdir())))

    snapshot_paths.extend(reversed(sorted(
        config.BRDF_CLASSIFIER_SNAPSHOTS_DIR.iterdir())))

    for snapshot_path in snapshot_paths:
        snapshot_name = snapshot_path.name
        checkpoint_dir = Path(config.BRDF_CLASSIFIER_DIR_REMOTE, 'checkpoints',
                              snapshot_name)
        if not checkpoint_dir.exists():
            continue
        model_json_paths = reversed(sorted(
            checkpoint_dir.glob('*/model_params.json')))
        for model_json_path in model_json_paths:
            model_name = model_json_path.parent.name
            with model_json_path.open('r') as f:
                model_params = json.load(f)
            best_stats_path = model_json_path.parent / 'model_best_stats.json'
            if best_stats_path.exists():
                with best_stats_path.open('r') as f:
                    best_stats = json.load(f)
            else:
                best_stats = None

            train_stats_path = model_json_path.parent / 'model_train_stats.json'
            if train_stats_path.exists():
                try:
                    with train_stats_path.open('r') as f:
                        train_stats = json.load(f)
                except json.JSONDecodeError:
                    train_stats = None
            else:
                train_stats = None

            snapshot_name = snapshot_path.name

            model_inference_dir = (config.BRDF_CLASSIFIER_DIR_REMOTE / 'inference'
                                   / snapshot_name / model_name)
            if model_inference_dir.exists():
                inference_epochs = [p.name for p in model_inference_dir.iterdir()]
            else:
                inference_epochs = []

            models_by_snapshot[snapshot_name].append({
                'params': model_params,
                'best_stats': best_stats,
                'train_stats': train_stats,
                'snapshot_name': snapshot_path.name,
                'model_name': model_json_path.parent.name,
                'model_path': str(model_json_path),
                'inference_epochs': inference_epochs,
            })

    return {
        'models_by_snapshot': models_by_snapshot,
    }


async def download_blend_file(request: web.Request):
    inference_path = request.query.get('inference_path', None)
    if not inference_path:
        raise web.HTTPNotFound()
    inference_path = Path(inference_path)
    if not inference_path.exists():
        raise web.HTTPNotFound()

    animate = request.query.get('animate', 'false') == 'true'
    no_floor = request.query.get('no_floor', 'false') == 'true'
    scene_type = request.query.get('type', 'inferred')
    if scene_type not in {'inferred', 'mtl'}:
        raise web.HTTPBadRequest(text="Invalid scene type.")

    filename = Path('/tmp', f'{uuid.uuid4()}.blend')

    command = [
        'python', '-m', 'terial.classifier.rendering.create_blend_file',
        str(inference_path), filename,
        '--pack-assets',
        '--type', scene_type,
        '--use-weighted-scores',
        # '--use-minc-substances',
    ]
    if animate:
        command.append('--animate')

    if no_floor:
        command.append('--no-floor')

    logger.info('Launching command %s', str(command))

    subprocess.call(command)
    if animate:
        out_name = f'{inference_path.stem}.{scene_type}.anim.blend'
    else:
        out_name = f'{inference_path.stem}.{scene_type}.blend'

    response = web.StreamResponse(
        status=200,
        headers=MultiDict({
            'Content-Disposition': f'attachment; filename={out_name}'
        }),
    )
    await response.prepare(request)
    with filename.open('rb') as f:
        await response.write(f.read())
    await response.write_eof()

    return response


@aiohttp_jinja2.template('classifier/inference_results_detailed.html')
async def show_inference_results_detailed(request):
    return await show_inference_results(request)


@aiohttp_jinja2.template('classifier/inference_results_simple.html')
async def show_inference_results_simple(request):
    return await show_inference_results(request)


async def show_inference_results(request: web.Request):
    snapshot_name = request.match_info.get('snapshot')
    model_name = request.match_info.get('model')
    epoch = request.match_info.get('epoch')
    page = int(request.query.get('page', 0))
    page_size = int(request.query.get('page_size', 12))
    topk = int(request.query.get('topk', 3))
    shape_ids = request.query.get('shape_ids', None)
    pair_ids = request.query.get('pair_ids', None)
    shape_source = request.query.get('shape_source', None)
    shape_category = request.query.get('shape_category', None)
    max_dist = float(request.query.get('max_dist', config.INFERENCE_MAX_DIST))
    if shape_ids is not None:
        shape_ids = shape_ids.replace(' ', '').strip(', ')
        shape_ids = [int(i) for i in shape_ids.split(',')]

    if pair_ids is not None:
        pair_ids = pair_ids.replace(' ', '').strip(', ')
        pair_ids = [int(i) for i in pair_ids.split(',')]

    if model_name is None:
        raise web.HTTPBadRequest()

    if snapshot_name is None:
        raise web.HTTPBadRequest()

    snapshot_dir = config.BRDF_CLASSIFIER_DIR_REMOTE / 'snapshots' / snapshot_name
    if not snapshot_dir.exists():
        snapshot_dir = config.BRDF_CLASSIFIER_DIR_REMOTE / 'lmdb' / snapshot_name
    if not snapshot_dir.exists():
        raise web.HTTPNotFound()

    inference_dir = (config.BRDF_CLASSIFIER_DIR_REMOTE / 'inference'
                     / snapshot_name / model_name / epoch)

    if not inference_dir.exists():
        raise web.HTTPNotFound()

    if (snapshot_dir / 'snapshot.json').exists():
        with (snapshot_dir / 'snapshot.json').open('r') as f:
            snapshot_dict = json.load(f)
    else:
        with (snapshot_dir / 'meta.json').open('r') as f:
            snapshot_dict = json.load(f)

    filters = []
    if shape_ids:
        filters.append(models.Shape.id.in_(shape_ids))

    if pair_ids:
        filters.append(models.ExemplarShapePair.id.in_(pair_ids))

    if shape_source:
        filters.append(models.Shape.source == shape_source)

    if shape_category:
        filters.append(models.Shape.category == shape_category)

    if topk:
        filters.append(models.ExemplarShapePair.rank <= topk)

    with session_scope() as sess:
        materials, _ = controllers.fetch_materials(sess)
        mat_by_id = {
            m.id: m for m in materials
        }
        shapes, count, pair_count = controllers.fetch_shapes_with_pairs(
            sess, page=page,
            page_size=page_size,
            filters=filters,
            max_dist=max_dist)

        results_by_shape = _aggregate_shape_inferences(
            inference_dir,
            mat_by_id,
            shapes, topk=topk, max_dist=max_dist)

    n_pages = int(math.ceil(count / page_size))

    return {
        'snapshot_name': snapshot_name,
        'dataset_name': snapshot_dict.get(
            'dataset_name', snapshot_dict.get('dataset', '<unknown dataset>')),
        'model_name': model_name,
        'epoch': epoch,
        'cur_page': page,
        'page_size': page_size,
        'n_total': count,
        'n_pairs': pair_count,
        'n_pages': n_pages,
        'results_by_shape': results_by_shape,
        'mat_by_id': mat_by_id,
    }


def _aggregate_shape_inferences(inference_dir,
                                mat_by_id, shapes,
                                topk, max_dist):
    results_by_shape = defaultdict(list)
    for shape in shapes:
        pairs = shape.get_topk_pairs(topk, max_dist)
        for pair in pairs:
            pair_inference = compute_pair_inference(
                inference_dir, mat_by_id, pair)
            if pair_inference is None:
                continue
            results_by_shape[pair.shape].append(pair_inference)
    return results_by_shape


def compute_pair_inference(inference_dir, mat_by_id, pair):
    # seg_substances = compute_segment_substances(pair, return_ids=True)
    seg_substances = None
    json_path = Path(inference_dir, f'{pair.id}.json')
    if not json_path.exists():
        logger.info('JSON path %s does not exist', json_path)
        return None

    with json_path.open('r') as f:
        inference_dict = json.load(f)

    try:
        seg_map = pair.load_data(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME) - 1
    except FileNotFoundError:
        seg_map = pair.load_data(config.PAIR_SHAPE_CLEAN_SEGMENT_MAP_NAME_OLD) - 1
    seg_mask_by_seg_id = {}
    for seg_id in np.unique(seg_map):
        if seg_id == -1:
            continue
        seg_mask_by_seg_id[str(seg_id)] = toolbox.web.image_to_base64(
            (seg_map == seg_id).astype(np.uint8) * 255)

    # _remove_disabled_materials(inference_dict, mat_by_id)
    compute_weighted_scores(inference_dict, mat_by_id, sort=True,
                            force_substances=True)

    rend_baseline_path = (config.BRDF_CLASSIFIER_DIR_REMOTE
                          / 'inference' / 'random' / 'renderings-cropped'
                          / f'{pair.id}.inferred.0000.jpg')
    rend_minc_inferred_path = (inference_dir.parent / '45-minc-subst'
                               / 'renderings-cropped'
                               / f'{pair.id}.inferred.0000.jpg')
    rend_inferred_path = (inference_dir / 'renderings-cropped'
                          / f'{pair.id}.inferred.0000.jpg')
    # if not rend_inferred_path.exists():
    #     rend_inferred_path = (inference_dir / 'renderings-aligned-cropped'
    #                           / f'{pair.id}.inferred.0000.jpg')
    # if not rend_inferred_path.exists():
    #     rend_inferred_path = (inference_dir / 'renderings-cropped'
    #                           / f'{pair.id}.inferred.0000.jpg')
    rend_inferred_frontal_path = (inference_dir / 'renderings-frontal'
                          / f'{pair.id}.inferred.0000.jpg')
    rend_mtl_path = (config.BRDF_CLASSIFIER_DIR_REMOTE
                          / 'inference' / 'default' / 'renderings-cropped'
                          / f'{pair.id}.mtl.0000.jpg')
    # rend_mtl_path = (inference_dir / 'renderings'
    #                  / f'{pair.id}.mtl.0000.png')

    rendering_path = inference_dir / 'renderings' / f'{pair.id}.png'

    return {
        'pair': pair,
        'shape': pair.shape,
        'exemplar': pair.exemplar,
        'seg_mask_by_seg_id': seg_mask_by_seg_id,
        'rend_inferred_path': rend_inferred_path,
        'rend_inferred_frontal_path': rend_inferred_frontal_path,
        'rend_minc_inferred_path': rend_minc_inferred_path,
        'rend_mtl_path': rend_mtl_path,
        'rend_baseline_path': rend_baseline_path,
        'rendering_path': rendering_path,
        'inference_path': json_path,
        **inference_dict,
    }


def _remove_disabled_materials(inference_dict, mat_by_id):
    for seg_id, seg_topk in inference_dict['segments'].items():
        seg_new_topk = []
        for match in seg_topk['material']:
            mat = mat_by_id[match['id']]
            if mat.enabled:
                seg_new_topk.append(match)
        seg_topk['material'] = seg_new_topk