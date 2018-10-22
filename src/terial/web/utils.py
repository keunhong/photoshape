import cv2
import io

import aiohttp
import json

import multidict
import numpy as np
from PIL import Image
from aiohttp import web


async def make_http_client():
    return aiohttp.ClientSession()


async def read_image_from_part(part: aiohttp.BodyPartReader):
    """
    Reads an image from a multipart message.
    """
    data = io.BytesIO(await part.read(decode=False))
    return np.asarray(Image.open(data))


async def read_exr_from_part(part: aiohttp.BodyPartReader):
    """
    Reads an image from a multipart message.
    """
    # data = io.BytesIO(await part.read(decode=False))
    # print(data)
    data = cv2.imdecode(np.asarray(await part.read()), flags=cv2.IMREAD_ANYDEPTH)
    return np.asarray(data)


def response_404(path='', headers=None, **kwargs):
    if path:
        message = 'Requested path ({path}) was not found'.format(path=path)
    else:
        message = 'Requested path was not found'
    if headers:
        headers = multidict.CIMultiDict(headers)
        headers.pop('Content-Type')
    return web.json_response({
        'status': 'error',
        'message': message,
    }, status=404, headers=headers, **kwargs)


def response_error(status, message, headers=None, **kwargs):
    if headers:
        headers = multidict.CIMultiDict(headers)
        headers.pop('Content-Type')
    return web.json_response({
        'status': 'error',
        'error': message,
    }, status=status, headers=headers, **kwargs)


def response_payload(payload, status=200, headers=None, **kwargs):
    if headers:
        headers = multidict.CIMultiDict(headers)
        headers.pop('Content-Type')
    return web.json_response({
        'status': 'success',
        'payload': payload,
    }, status=status, dumps=json.dumps, headers=headers, **kwargs)
