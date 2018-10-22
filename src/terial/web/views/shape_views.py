import math
from typing import List

import aiohttp_jinja2
import structlog
from aiohttp import web
import sqlalchemy as sa

from terial import models
from terial.database import session_scope
from terial.models import Shape


logger = structlog.get_logger(__name__)


@aiohttp_jinja2.template('list_shapes.html')
async def list_shapes(request: web.Request):
    page = int(request.query.get('page', 0))
    page_size = int(request.query.get('page_size', 100))
    source = request.query.get('source', None)

    filters = []
    if source:
        filters.append(Shape.source == source)

    with session_scope() as sess:
        query = sess.query(Shape).filter(sa.and_(*filters))
        shapes: List[Shape] = (query
                               .order_by(Shape.id.asc())
                               .offset(page * page_size)
                               .limit(page_size)
                               .all())

        count = query.count()

    n_pages = int(math.ceil(count / page_size))

    return {
        'shapes': shapes,
        'cur_page': page,
        'page_size': page_size,
        'n_total': count,
        'n_pages': n_pages,
    }


async def set_exclude(request: web.Request):
    shape_id = request.match_info.get('shape_id')
    if shape_id is None:
        raise web.HTTPBadRequest(text='Shape ID not given')

    exclude = (await request.post()).get('exclude')
    if exclude is None or exclude not in {'true', 'false'}:
        raise web.HTTPBadRequest(text='Invalid value for exclude')

    with session_scope() as sess:
        shape = sess.query(models.Shape).get(int(shape_id))
        if shape is None:
            raise web.HTTPNotFound(text='Shape not found')
        shape.exclude = exclude == 'true'
        sess.commit()

        return web.json_response({
            'status': 'success',
            'shape': shape.serialize(),
        })
