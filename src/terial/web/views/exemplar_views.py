import math
from typing import List

import aiohttp_jinja2
from aiohttp import web
import sqlalchemy as sa

from terial.database import session_scope
from terial.models import Exemplar


@aiohttp_jinja2.template('list_exemplars.html')
async def list_exemplars(request: web.Request):
    page = int(request.query.get('page', 0))
    page_size = int(request.query.get('page_size', 100))
    excluded = request.query.get('excluded', None)

    filters = []
    if excluded:
        filters.append(Exemplar.exclude == (excluded.lower() == 'true'))

    with session_scope() as sess:
        query = (sess.query(Exemplar)
                 .filter(sa.and_(*filters))
                 .order_by(Exemplar.id.asc())
                 )
        exemplars: List[Exemplar] = (query
                                     .offset(page * page_size)
                                     .limit(page_size)
                                     .all())
        count = query.count()

    n_pages = int(math.ceil(count / page_size))

    query = {}
    if excluded:
        query['excluded'] = excluded

    return {
        'exemplars': exemplars,
        'cur_page': page,
        'page_size': page_size,
        'n_total': count,
        'n_pages': n_pages,
        'query': query,
    }


async def set_exclude(request: web.Request):
    exemplar_id = request.match_info.get('exemplar_id')
    if exemplar_id is None:
        raise web.HTTPBadRequest(text='Exemplar ID not given')

    exclude = (await request.post()).get('exclude')
    if exclude is None or exclude not in {'true', 'false'}:
        raise web.HTTPBadRequest(text='Invalid value for exclude')

    with session_scope() as sess:
        exemplar = sess.query(Exemplar).get(int(exemplar_id))
        if exemplar is None:
            raise web.HTTPNotFound(text='Exemplar not found')
        exemplar.exclude = exclude == 'true'
        sess.commit()

        return web.json_response({
            'status': 'success',
            'exemplar': exemplar.serialize(),
        })
