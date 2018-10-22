import logging
from collections import Counter
from datetime import datetime

import aiohttp_jinja2

from aiohttp import web
import sqlalchemy as sa

from terial import config, controllers
from terial.controllers import fetch_random_shape_with_pair
from terial.database import session_scope
from terial.models import (ResultSet, ResultAnnotation, ExemplarShapePair,
                           Shape,
                           ResultQuality)
from terial.web.template_funcs import nginx_url

logger = logging.getLogger(__name__)


tokens = {
    'kpar': '36ee2879-ff99-4e28-8c64-45f36650b2e9',
    'krematas': 'cdb94bce-b70b-4dcb-b193-7f5e238e6a9e',
    'xuan': '6e8a07b3-50ab-4486-af01-56de8cd82133',
    'james': 'eda36b75-7cdb-4125-acb3-d14fca5aad2f',
    'edzhang': '8cefad16-10ed-4411-89b1-6d98edd0f71e',
    'joseph': '1556d25a-a193-4269-aa82-9a767880cd09',
    'junha': '6911af50-ec9c-47eb-8b29-8c833ec81952',
    'isaac': '402efd59-c0da-4a62-9b25-91c32e185583',
    'aditya': '9a43d3ac-6a0d-4368-ba26-14db883555e8',
    'jjpark': '62deb890-426f-4c3d-bc89-66eb13ca1d30',
}


def check_auth(query):
    username = query.get('username', None)
    if username is None:
        raise web.HTTPBadRequest()

    if username not in tokens:
        raise web.HTTPUnauthorized()

    token = query.get('token', None)
    if token is None:
        raise web.HTTPBadRequest()

    if token != tokens.get(username):
        raise web.HTTPUnauthorized()


def get_result_set_or_raise(query):
    result_set_id = query.get('result_set')
    if result_set_id is None:
        raise web.HTTPBadRequest(text="Must provide result set")
    try:
        result_set_id = int(result_set_id)
    except ValueError:
        raise web.HTTPBadRequest(text="Invalid result set ID")

    with session_scope() as sess:
        result_set = sess.query(ResultSet).get(result_set_id)
    if result_set is None:
        raise web.HTTPNotFound(text="No such result set")
    return result_set


@aiohttp_jinja2.template('evaluation/stats.html')
async def show_stats(request: web.Request):
    result_set_id = request.query.get('result_set')
    if result_set_id is None:
        raise web.HTTPBadRequest(text="Must provide result set")
    exclude_users = request.query.get('exclude_users', None)
    shape_source = request.query.get('shape_source', 'all')
    if exclude_users:
        exclude_users = set(exclude_users.split(','))

    topk = request.query.get('topk', config.INFERENCE_TOPK)
    max_dist = request.query.get('max_dist', config.INFERENCE_MAX_DIST)

    shape_annotations = {}
    pair_annotations = {}
    with session_scope() as sess:
        # Shape stats.
        shapes = controllers.fetch_shapes_with_annotations(
            sess, result_set_id, shape_source)
        for shape in shapes:
            annotations = [
                a for a in shape.result_annotations
                if ((a.quality not in {ResultQuality.bad_shape,
                                       ResultQuality.bad_exemplar,
                                       ResultQuality.not_sure})
                    and (exclude_users is None or
                         a.username not in exclude_users))]
            if len(annotations) == 0:
                continue
            else:
                shape_annotations[shape.id] = min(annotations).quality

        # Pair stats.
        pairs = controllers.fetch_pairs_with_annotations(
            sess, result_set_id, shape_source,
            filters=[ExemplarShapePair.distance <= float(max_dist),
                     ExemplarShapePair.rank <= int(topk)])
        for pair in pairs:
            annotations = [
                a for a in pair.result_annotations
                if ((a.quality not in {ResultQuality.bad_shape,
                                       ResultQuality.bad_exemplar,
                                       ResultQuality.not_sure})
                    and (exclude_users is None or
                         a.username not in exclude_users))]
            if len(annotations) == 0:
                continue
            else:
                pair_annotations[pair.id] = min(annotations).quality

    shape_quality_counts = dict(Counter(shape_annotations.values()))
    shape_quality_total = sum(shape_quality_counts.values())

    pair_quality_counts = dict(Counter(pair_annotations.values()))
    pair_quality_total = sum(pair_quality_counts.values())

    return {
        'shape_annotations': shape_annotations,
        'shape_quality_counts': shape_quality_counts,
        'shape_quality_total': shape_quality_total,

        'pair_annotations': pair_annotations,
        'pair_quality_counts': pair_quality_counts,
        'pair_quality_total': pair_quality_total,

        'ResultQuality': ResultQuality,
    }


@aiohttp_jinja2.template('evaluation/landing.html')
async def landing_page(request: web.Request):
    check_auth(request.query)

    result_set_id = request.query.get('result_set')
    if result_set_id is None:
        raise web.HTTPBadRequest(text="Must provide result set")
    shape_source = request.query.get('shape_source', 'all')

    return {
        'result_set': result_set_id,
        'shape_source': shape_source
    }


@aiohttp_jinja2.template('evaluation/main.html')
async def annotate(request: web.Request):
    check_auth(request.query)

    result_set_id = request.query.get('result_set')
    if result_set_id is None:
        raise web.HTTPBadRequest(text="Must provide result set")
    shape_source = request.query.get('shape_source', 'all')

    return {
        'result_set': result_set_id,
        'shape_source': shape_source
    }


async def get_job(request: web.Request):
    check_auth(request.query)
    shape_source = request.query.get('shape_source', 'all')
    result_set = get_result_set_or_raise(request.query)
    shape_id = request.query.get('shape_id', None)

    with session_scope() as sess:
        pair_dicts = []
        for attempts in range(50):
            if shape_id:
                shape = sess.query(Shape).get(int(shape_id))
                if shape is None:
                    raise web.HTTPNotFound()
            else:
                logger.info('@@@@ %s', shape_source)
                shape = fetch_random_shape_with_pair(
                    sess,
                    username=request.query['username'],
                    shape_source=shape_source)
                if shape is None:
                    return web.json_response(data={
                        'error': 'Looks like you are done!'
                    }, status=400)

            pairs = shape.get_topk_pairs(config.INFERENCE_TOPK,
                                         config.INFERENCE_MAX_DIST)
            for pair in pairs:
                rend_path = result_set.get_rendering_path(pair.id)
                if rend_path is None or not rend_path.exists():
                    continue
                pair_dicts.append({
                    'pair_id': pair.id,
                    'rend_url': nginx_url(rend_path),
                    'exemplar_url': nginx_url(pair.exemplar.cropped_path)
                })
            if len(pair_dicts) > 0:
                break
        if len(pair_dicts) == 0:
            raise web.HTTPInternalServerError(text="Could not find pairs..")

    return web.json_response(data={
        'shape': shape.serialize(),
        'pairs': pair_dicts,
    })


async def post_annotation(request: web.Request):
    data = dict(await request.post())
    check_auth(data)
    result_set = get_result_set_or_raise(data)
    try:
        pair_id = int(data['pair_id'])
    except ValueError:
        raise web.HTTPBadRequest(text="Invalid pair_id")

    with session_scope() as sess:
        pair = sess.query(ExemplarShapePair).get(pair_id)
        if pair is None:
            raise web.HTTPNotFound(text="No such pair")

        annotation = sess.query(ResultAnnotation).filter_by(
            username=data['username'],
            shape_id=pair.shape.id,
            pair_id=pair.id,
            result_set_id=result_set.id
        ).first()

        if annotation:
            annotation.category = data['category']
        else:
            annotation = ResultAnnotation(
                username=data['username'],
                category=data['category'],
                shape_id=pair.shape.id,
                pair_id=pair.id,
                result_set_id=result_set.id,
                date_updated=datetime.now()
            )
            sess.add(annotation)
        sess.commit()

        return web.json_response(data=annotation.serialize())
