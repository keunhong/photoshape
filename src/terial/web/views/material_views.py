import json
from functools import partial

from collections.__init__ import defaultdict
from typing import List

import aiohttp_jinja2
import math
import sqlalchemy as sa
from aiohttp import web

from terial import controllers
from terial.database import session_scope
from terial.models import Material
from terial.web.template_funcs import nginx_url


SUBSTANCES = {'fabric', 'wood', 'leather', 'metal', 'plastic'}


@aiohttp_jinja2.template('list_materials.html')
async def list_materials(request: web.Request):
    substances = request.query.get('substance', None)
    types = request.query.get('type', None)
    page = int(request.query.get('page', 0))
    page_size = int(request.query.get('page_size', 25))
    filters = []
    if substances:
        filters.append(Material.substance.in_(substances.split(',')))
    if types:
        filters.append(Material.type.in_(types.split(',')))

    with session_scope() as sess:
        materials, count = controllers.fetch_materials(
            sess, page_size, page,
            order_by=(Material.substance.asc(), Material.id.asc()),
            filters=filters)

    n_pages = int(math.ceil(count / page_size))

    return {
        'query': request.query,
        'substances': SUBSTANCES,
        'materials': materials,
        'cur_page': page,
        'page_size': page_size,
        'n_total': count,
        'n_pages': n_pages,
    }


@aiohttp_jinja2.template('material_tree.html')
async def material_tree(request: web.Request):
    return {}


async def material_tree_json(request: web.Request):
    with session_scope() as sess:
        materials: List[Material] = (
            sess.query(Material)
                .filter(Material.enabled.is_(True))
                .order_by(Material.substance.asc(), Material.id.asc())
                .all())

    material_by_substance = defaultdict(list)
    for material in materials:
        if material.substance:
            material_by_substance[material.substance].append(material)

    tree = {
        'name': 'materials',
        'children': [{
            'name': subst,
            'text': subst,
            'children': [
                {
                    'name': mat.name,
                    'size': 120000,
                    'img': nginx_url(mat.get_data_path('previews/bmps.png')),
                }
                for mat in subst_mats if mat.substance]
        } for subst, subst_mats in material_by_substance.items()],
    }

    return web.json_response(tree, dumps=partial(json.dumps, indent=2))


async def set_default_scale(request: web.Request):
    material_id = request.match_info.get('material_id')
    if material_id is None:
        raise web.HTTPBadRequest(text='Material ID not given')

    default_scale = (await request.post()).get('default_scale')
    if default_scale is None:
        raise web.HTTPBadRequest(text='default_scale not given')
    try:
        default_scale = float(default_scale)
    except ValueError:
        raise web.HTTPBadRequest(text='default_scale must be a float')

    with session_scope() as sess:
        material = sess.query(Material).get(int(material_id))
        if material is None:
            raise web.HTTPNotFound(text='Material not found')
        material.default_scale = default_scale
        sess.commit()

        return web.json_response({
            'status': 'success',
            'material': material.serialize(),
        })


async def set_enabled(request: web.Request):
    material_id = request.match_info.get('material_id')
    if material_id is None:
        raise web.HTTPBadRequest(text='Material ID not given')

    enabled = (await request.post()).get('enabled')
    if enabled is None or enabled not in {'true', 'false'}:
        raise web.HTTPBadRequest(text='Invalid value for enabled')

    with session_scope() as sess:
        material = sess.query(Material).get(int(material_id))
        if material is None:
            raise web.HTTPNotFound(text='Material not found')
        material.enabled = enabled == 'true'
        sess.commit()

        return web.json_response({
            'status': 'success',
            'material': material.serialize(),
        })
