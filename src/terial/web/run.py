import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp_jinja2
import jinja2
from aiohttp import web
from aiohttp.web_middlewares import normalize_path_middleware

from terial import config
from terial.web import template_funcs, routes
from terial.web.middleware import prefix_middleware, common_exception_middleware

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def merge_dicts(*args):
    out = {}
    for d in args:
        out.update(d)
    return out


def is_dict(d):
    return isinstance(d, dict)


def main():
    loop = asyncio.get_event_loop()

    prefix = '/terial'

    app = web.Application(middlewares=[
        normalize_path_middleware(),
        prefix_middleware(prefix),
        common_exception_middleware(),
    ])
    aiohttp_jinja2.setup(
        app,
        loader=jinja2.FileSystemLoader(str(script_dir / 'templates')),
        context_processors=[
            aiohttp_jinja2.request_processor
        ],
    )
    jinja_env = aiohttp_jinja2.get_env(app)
    template_funcs.setup(jinja_env)
    jinja_env.globals['config'] = config
    jinja_env.globals['merge_dicts'] = merge_dicts
    jinja_env.globals['is_dict'] = is_dict

    routes.setup(app, prefix)

    app['executor'] = ThreadPoolExecutor(max_workers=3)

    web.run_app(app, port=9999, print=logger.info)


if __name__ == '__main__':
    main()
