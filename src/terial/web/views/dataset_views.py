import math
import ujson as json
from pathlib import Path

import aiocache as aiocache
import aiohttp_jinja2
from aiohttp import web

import sqlalchemy as sa
from sqlalchemy import orm

from terial import config, models
from terial.database import session_scope


@aiocache.cached(ttl=10)
async def _get_rend_paths(dataset_name):
    dataset_dir = config.BRDF_CLASSIFIER_DATASETS_DIR / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    paths = []
    for client_dir in dataset_dir.iterdir():
        for epoch_dir in client_dir.iterdir():
            for split_dir in epoch_dir.iterdir():
                paths.extend(split_dir.glob('*.ldr.png'))
    return paths


@aiohttp_jinja2.template('classifier/list_datasets.html')
async def list_datasets(request: web.Request):
    dataset_names = [o.name
                     for o in config.BRDF_CLASSIFIER_DATASETS_DIR.iterdir()]

    return {
        'dataset_names': dataset_names,
        'n_total': len(dataset_names),
    }


@aiohttp_jinja2.template('classifier/show_dataset_renderings.html')
async def search_dataset_renderings(request: web.Request):
    dataset_name = request.match_info.get('dataset_name', '')
    page = int(request.query.get('page', 0))
    page_size = int(request.query.get('page_size', 100))

    material_id = request.query.get('material_id', None)
    try:
        max_saturated_frac = float(
            request.query.get('max_saturated_frac', 0.05))
    except ValueError:
        raise web.HTTPBadRequest(text='max_saturated_frac must be a float')

    try:
        min_saturated_frac = float(
            request.query.get('min_saturated_frac', 0.0))
    except ValueError:
        raise web.HTTPBadRequest(text='min_saturated_frac must be a float')

    filters = [
        models.Rendering.dataset_name == dataset_name,
        models.Rendering.saturated_frac <= max_saturated_frac,
        models.Rendering.saturated_frac >= min_saturated_frac,
    ]

    if material_id:
        try:
            material_id = int(material_id)
        except ValueError:
            raise web.HTTPBadRequest(text='material_id must be integer')
        else:
            filters.append(models.Material.id == material_id)

    with session_scope() as sess:
        query = (sess.query(models.Rendering)
                 .filter_by(exclude=False)
                 .join((models.Material, models.Rendering.materials))
                 .filter(sa.and_(*filters))
                 .group_by(models.Rendering.id))
        num_total = query.count()
        renderings = query.offset(page * page_size).limit(page_size)

    num_pages = int(math.ceil(num_total / page_size))

    query = {}
    if material_id:
        query['material_id'] = material_id

    return {
        'renderings': renderings,
        'cur_page': page,
        'page_size': page_size,
        'n_total': num_total,
        'n_pages': num_pages,
        'dataset_name': dataset_name,
        'query': query,
    }


@aiohttp_jinja2.template('classifier/list_snapshots.html')
async def list_snapshots(request: web.Request):
    snapshot_names = [o.name for o in config.BRDF_CLASSIFIER_SNAPSHOTS_DIR.iterdir()]
    snapshot_names.sort(key=lambda o: o)

    return {
        'snapshot_names': snapshot_names,
        'n_total': len(snapshot_names),
    }
