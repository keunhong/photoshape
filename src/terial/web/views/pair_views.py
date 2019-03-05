import io
import math

import aiohttp_jinja2
from PIL import Image
from aiohttp import web
from multidict import MultiDict
from sqlalchemy import orm

from terial import controllers, config
from terial.database import session_scope
from terial.models import ExemplarShapePair, Shape
from terial.pairs import utils as pair_utils


async def _list_pairs(request):
    by_shape = request.query.get('by_shape', 'true') == 'true'
    max_dist = request.query.get('max_dist', config.ALIGN_DIST_THRES)
    source = request.query.get('source', None)
    sort_field = request.query.get('sort_field', 'distance')
    sort_order = request.query.get('sort_order', 'asc')
    category = request.query.get('category', None)
    shape_id = request.query.get('shape_id', None)
    sort_field = {
        'pair_id': ExemplarShapePair.id,
        'exemplar_id': ExemplarShapePair.exemplar_id,
        'shape_id': ExemplarShapePair.shape_id,
        'distance': ExemplarShapePair.distance,
    }.get(sort_field, None)
    pair_ids = request.query.get('pair_ids', None)

    if sort_field is None:
        return {
            'error': 'Invalid value for sort_field',
        }
    if sort_order == 'asc':
        order_by = sort_field.asc()
    elif sort_order == 'desc':
        order_by = sort_field.desc()
    else:
        return {
            'error': 'Invalid value for sort_order',
        }

    filters = []
    if pair_ids:
        pair_ids = [int(i) for i in pair_ids.split(',')]
        filters.append(ExemplarShapePair.id.in_(pair_ids))

    if source:
        filters.append(ExemplarShapePair.shape.has(source=source))
    if category is not None:
        filters.append(ExemplarShapePair.shape.has(category=category))
    if shape_id is not None:
        filters.append(ExemplarShapePair.shape.has(id=shape_id))

    consistent_segments = \
        request.query.get('consistent_segments', 'false') == 'true'
    if consistent_segments:
        filters.append(
            ExemplarShapePair.num_segments >= ExemplarShapePair.num_substances)
        filters.append(
            ExemplarShapePair.num_segments.isnot(None))

    page = int(request.query.get('page', 0))
    page_size = int(request.query.get('page_size', 100))
    by_shape_topk = int(request.query.get('by_shape_topk', 5))

    with session_scope() as sess:
        pairs, count = controllers.fetch_pairs(
            sess,
            by_shape=by_shape,
            by_shape_topk=by_shape_topk,
            max_dist=max_dist,
            filters=filters,
            page_size=page_size,
            page=page,
            order_by=order_by)

    n_pages = int(math.ceil(count / page_size))

    return {
        'pairs': pairs,
        'cur_page': page,
        'page_size': page_size,
        'n_total': count,
        'n_pages': n_pages,
    }


@aiohttp_jinja2.template('list_pairs.html')
async def list_pairs(request: web.Request):
    return await _list_pairs(request)


@aiohttp_jinja2.template('list_pairs_kostas.html')
async def list_pairs_kostas(request: web.Request):
    return await _list_pairs(request)


@aiohttp_jinja2.template('show_pair.html')
async def show_pair(request: web.Request):
    pair_id = request.match_info.get('pair_id')

    with session_scope() as sess:
        pair = (sess.query(ExemplarShapePair)
                .options(orm.joinedload(ExemplarShapePair.exemplar),
                         orm.joinedload(ExemplarShapePair.shape))
                .get(int(pair_id)))

    if pair is None:
        raise web.HTTPNotFound()

    return {
        'pair': pair,
    }


async def get_uncropped_exemplar(request: web.Request):
    pair_id = request.match_info['pair_id']
    with session_scope() as sess:
        pair = sess.query(ExemplarShapePair).get(pair_id)
        if pair is None:
            raise web.HTTPNotFound(text='No such pair.')

        uncropped_im = pair_utils.pair_uncropped_exemplar(pair)

    image = Image.fromarray(uncropped_im)
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    buf.seek(0)

    out_name = f'{pair.id}.{pair.exemplar.id}.uncropped.jpg'
    response = web.StreamResponse(
        status=200,
        headers=MultiDict({
            'Content-Type': 'image/jpeg',
            # 'Content-Disposition': f'attachment; filename={out_name}'
        }))
    await response.prepare(request)
    await response.write(buf.getvalue())
    await response.write_eof()
    return response
