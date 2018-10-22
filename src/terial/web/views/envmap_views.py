from typing import List

import aiohttp_jinja2
from aiohttp import web

from terial import models
from terial.database import session_scope


@aiohttp_jinja2.template('list_envmaps.html')
async def list_envmaps(request: web.Request):
    with session_scope() as sess:
        envmaps: List[models.Envmap] = (
            sess.query(models.Envmap)
                .order_by(models.Envmap.enabled.desc(),
                          models.Envmap.split_set.asc(),
                          models.Envmap.id.asc())
                .all())

    return {
        'envmaps': envmaps
    }