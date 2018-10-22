import json
import math

import aiohttp_jinja2
import random
from aiohttp import web
import sqlalchemy as sa
from sqlalchemy import func

from terial import config, models, controllers
from terial.database import session_scope
from terial.models import ExemplarShapePair
from terial.web.views.classifier_views import (show_inference_results,
                                               compute_pair_inference)


@aiohttp_jinja2.template('figures/materials.html')
async def show_all_materials(request):
    with session_scope() as sess:
        materials = sess.query(models.Material).filter_by(enabled=True).all()

    return {
        'materials': materials
    }


@aiohttp_jinja2.template('figures/exemplars.html')
async def show_exemplars(request):
    page = int(request.query.get('page', 0))
    page_size = int(request.query.get('page_size', 100))
    with session_scope() as sess:
        exemplars = sess.query(models.Exemplar).filter_by(exclude=False).offset(page*page_size).limit(page_size).all()

    return {
        'exemplars': exemplars
    }


@aiohttp_jinja2.template('figures/shapes.html')
async def show_shapes(request):
    page = int(request.query.get('page', 0))
    page_size = int(request.query.get('page_size', 100))
    with session_scope() as sess:
        shapes = sess.query(models.Shape).filter_by(exclude=False).offset(page*page_size).limit(page_size).all()

    return {
        'shapes': shapes,
    }


@aiohttp_jinja2.template('figures/prcs.html')
async def show_prcs(request):
    resultset_id = request.match_info.get('resultset_id')

    with session_scope() as sess:
        resultset = sess.query(models.ResultSet).get(resultset_id)
        snapshot_name = resultset.snapshot_name
        model_name = resultset.model_name
        epoch = resultset.inference_name

    page = int(request.query.get('page', 0))
    page_size = int(request.query.get('page_size', 100))
    shuffle = request.query.get('shuffle', 'false') == 'true'
    max_dist = int(request.query.get('max_dist', config.INFERENCE_MAX_DIST))
    topk = int(request.query.get('topk', config.INFERENCE_TOPK))

    pair_ids = request.query.get('pair_ids', None)
    if pair_ids is not None:
        pair_ids = pair_ids.replace(' ', '').strip(', ')
        pair_ids = [int(i) for i in pair_ids.split(',')]

    filters = []
    if pair_ids:
        filters.append(models.ExemplarShapePair.id.in_(pair_ids))

    if shuffle:
        order_by = func.random()
    else:
        order_by = ExemplarShapePair.shape_id.asc()

    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs(
            sess, page=page, page_size=page_size,
            max_dist=max_dist,
            order_by=order_by,
            by_shape_topk=topk,
            by_shape=True,
            filters=filters)

    n_pages = int(math.ceil(count / page_size))

    inference_dir = (config.BRDF_CLASSIFIER_DIR_REMOTE / 'inference'
                     / snapshot_name / model_name / epoch)
    print(inference_dir)

    return {
        'inference_dir': inference_dir,
        'snapshot_name': snapshot_name,
        'model_name': model_name,
        'epoch': epoch,
        'cur_page': page,
        'page_size': page_size,
        'n_total': count,
        'n_pages': n_pages,
        'pairs': pairs,
        'resultset_id': resultset_id,
    }


@aiohttp_jinja2.template('figures/inference_results_figure.html')
async def show_inference_results_figure(request):
    return await show_inference_results(request)


async def _pair_inference_results(request):
    snapshot_name = request.match_info.get('snapshot')
    model_name = request.match_info.get('model')
    epoch = request.match_info.get('epoch')
    page = int(request.query.get('page', 0))
    page_size = int(request.query.get('page_size', 100))
    num_cols = int(request.query.get('num_cols', 1))

    category = request.query.get('category')

    pair_ids = request.query.get('pair_ids', None)
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
    if pair_ids:
        filters.append(models.ExemplarShapePair.id.in_(pair_ids))

    if category is not None:
        filters.append(ExemplarShapePair.shape.has(category=category))

    with session_scope() as sess:
        materials, _ = controllers.fetch_materials(sess)
        mat_by_id = {
            m.id: m for m in materials
        }
        pairs, count = controllers.fetch_pairs(
            sess, page=page, page_size=page_size,
            max_dist=config.INFERENCE_MAX_DIST,
            by_shape_topk=3,
            by_shape=True,
            filters=filters)

        pair_inferences = []
        for pair in pairs:
            pair_inference = get_pair_inference_figures(inference_dir, pair)
            if pair_inference is not None:
                pair_inferences.append(pair_inference)

    # random.shuffle(pair_inferences)
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
        'n_pages': n_pages,
        'pair_inferences': pair_inferences,
        'mat_by_id': mat_by_id,
        'num_cols': num_cols,
        'col_size': math.ceil(len(pair_inferences) / num_cols),
    }

@aiohttp_jinja2.template('figures/inference_results_figure_2.html')
async def show_inference_results_figure_2(request):
    return await _pair_inference_results(request)


@aiohttp_jinja2.template('figures/renderings.html')
async def show_renderings(request: web.Request):
    dataset_name = 'raw-20180426'
    page = int(request.query.get('page', 0))
    page_size = int(request.query.get('page_size', 100))

    with session_scope() as sess:
        query = (sess.query(models.Rendering)
                 .filter_by(exclude=False)
                 .join((models.Material, models.Rendering.materials))
                 .group_by(models.Rendering.id))
        num_total = query.count()
        renderings = query.offset(page * page_size).limit(page_size)

    num_pages = int(math.ceil(num_total / page_size))

    query = {}

    return {
        'renderings': renderings,
        'cur_page': page,
        'page_size': page_size,
        'n_total': num_total,
        'n_pages': num_pages,
        'dataset_name': dataset_name,
        'query': query,
    }


def get_pair_inference_figures(inference_dir, pair):
    rend_baseline_path = (config.BRDF_CLASSIFIER_DIR_REMOTE
                          / 'inference' / 'random' / 'renderings-cropped'
                          / f'{pair.id}.inferred.0000.jpg')
    rend_minc_inferred_path = (inference_dir.parent / '45-minc-subst'
                               / 'renderings-cropped'
                               / f'{pair.id}.inferred.0000.jpg')
    rend_inferred_path = (inference_dir / 'renderings-aligned-cropped'
                          / f'{pair.id}.inferred.0000.jpg')
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
        'rend_inferred_path': rend_inferred_path,
        'rend_inferred_frontal_path': rend_inferred_frontal_path,
        'rend_minc_inferred_path': rend_minc_inferred_path,
        'rend_mtl_path': rend_mtl_path,
        'rend_baseline_path': rend_baseline_path,
        'rendering_path': rendering_path,
    }
