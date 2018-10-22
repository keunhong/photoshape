import asyncio
from aiohttp import web
from aiohttp.web_exceptions import _HTTPMove
import structlog

from terial.web.utils import response_error, response_404

logger = structlog.get_logger(__name__)


def prefix_middleware(prefix):

    async def middleware(app, handler):

        async def middleware_handler(request):
            response = await handler(request)
            if isinstance(response, _HTTPMove):
                prefix = request.headers.get('X-Path-Prefix', '')
                response.location = prefix + response.location
                # response.location.replace(prefix, '/')
                response.location.replace('//', '/')
                response.headers['Location'] = str(
                    prefix + response.headers['Location'])
            return response

        return middleware_handler

    return middleware


def common_exception_middleware():
    async def middleware(app, handler):
        async def middleware_handler(request):
            try:
                response = await handler(request)
                return response
            except web.HTTPNotFound as e:
                return response_404(path=request.path, headers=e.headers)
            except web.HTTPException as e:
                return response_error(e.status_code, e.text,
                                      headers=e.headers)
            except asyncio.CancelledError:
                raise
            # except Exception as e:
            #     logger.exception(
            #         f'unexpected exception ({e.__class__.__qualname__}) '
            #         f'in request handler', handler=handler, request=request)
            #     return response_error(500, str(e))
        return middleware_handler
    return middleware
