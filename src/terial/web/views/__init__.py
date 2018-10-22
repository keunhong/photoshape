import aiohttp_jinja2
from aiohttp import web

from terial.web.views.dataset_views import search_dataset_renderings
from terial.web.views.envmap_views import list_envmaps
from terial.web.views.pair_views import list_pairs
from terial.web.views.shape_views import list_shapes
from terial.web.views.material_views import list_materials
from terial.web.views.exemplar_views import list_exemplars
from toolbox.logging import init_logger

logger = init_logger(__name__)


@aiohttp_jinja2.template('index.html')
async def index(request: web.Request):
    return {}
