import argparse
import os
import re
import skimage.transform
from functools import partial
from pathlib import Path

import aiohttp

import numpy as np
import json
import click
import structlog
import visdom
import warnings
from aiohttp import web
from aiohttp.web_middlewares import normalize_path_middleware
from skimage.io import imsave

from terial import models
from terial.classifier.data.utils import compute_saturated_frac
from terial.database import session_scope
from terial.web.utils import read_image_from_part, read_exr_from_part

vis = visdom.Visdom(env='data-collector')
logger = structlog.get_logger()


parser = argparse.ArgumentParser()
parser.add_argument('--host', default='localhost')
parser.add_argument('--port', default=9500)
parser.add_argument('--exr-out', type=Path)
parser.add_argument('out', type=Path)
args = parser.parse_args()
dataset_name = args.out.name

shapes = [(384, 384), (500, 500)]


def safe_string(s):
    """
    Turns string into a string safe for filenames.
    """
    return "".join(c for c in s if re.match(r'\w|-|_|.', c))


def imshow(image, filename):
    """
    Sends image to visdom.
    """
    if len(image.shape) == 3:
        vis.image(image.transpose((2, 0, 1)), win=filename,
                  opts={'title': filename})
    else:
        vis.image(image, win=filename,
                  opts={'title': filename})


async def hello(request: web.Request) -> web.Response:
    return web.json_response(data={'status': 'success'})


async def data_handler(request: web.Request) -> web.Response:
    reader = await request.multipart()
    part = await reader.next()
    if part.filename == 'metadata.json':
        metadata = await part.json()
    else:
        return web.json_response(data={
            'status': 'error',
            'error': 'metadata must come first',
        })

    try:
        client_id = safe_string(metadata['client_id'])
        pair_id = int(metadata['pair_id'])
        epoch = int(metadata['epoch'])
        iteration = int(metadata['iteration'])
        split_set = metadata['split_set']
        prefix = f'{pair_id:05d}_{iteration:03d}'
    except (KeyError, ValueError) as e:
        logger.exception('Error while parsing metadata')
        return web.json_response(
            data={
                'status': 'error',
                'error': f'Invalid input: {str(e)}',
            }
        )

    files = {}

    async for part in reader:
        out_dir = Path(args.out,
                       f'client={client_id}', f'epoch={epoch:03d}',
                       split_set)
        exr_out_dir = args.exr_out if args.exr_out else out_dir

        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        if not exr_out_dir.exists():
            exr_out_dir.mkdir(parents=True)

        out_path = Path(out_dir, f'{prefix}.{part.filename}')

        # Handle contents.
        content_type = part.headers[aiohttp.hdrs.CONTENT_TYPE]
        if content_type == 'image/png' or content_type == 'image/jpeg':
            image = await read_image_from_part(part)

            files[part.filename] = (out_path, content_type, image)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(str(out_path), image)
                for shape in shapes:
                    if 'map.' in part.filename:
                        image_resized = skimage.transform.resize(
                            image, shape, order=0, anti_aliasing=False,
                            mode='constant', preserve_range=True,
                            cval=0).astype(np.uint8)
                    else:
                        image_resized = skimage.transform.resize(
                            image, shape, order=1, anti_aliasing=True,
                            mode='reflect')
                    base = os.path.splitext(part.filename)[0]
                    ext = out_path.suffix
                    resized_path = Path(
                        out_dir, f'{prefix}.{base}.{shape[0]}x{shape[1]}{ext}')
                    imsave(str(resized_path), image_resized)
            imshow(image, filename=part.filename)
        elif content_type == 'image/x-exr':
            exr_out_path = Path(exr_out_dir, f'{prefix}.{part.filename}')
            bytes = await part.read(decode=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(out_path, 'wb') as f:
                    f.write(bytes)
        elif content_type == 'application/json':
            data = await part.json()
            files[part.filename] = (out_path, content_type, data)
        else:
            logger.error('Unsupported content type',
                         content_type=content_type,
                         filename=part.filename)

    _, _, ldr_image = files['ldr.jpg']
    _, _, segment_map = files['segment_map.png']

    saturated_frac = compute_saturated_frac(ldr_image, segment_map > 0)

    for filename, (out_path, content_type, content) in files.items():
        if filename == 'params.json':
            content['saturated_frac'] = saturated_frac

        if content_type == 'application/json':
            with open(out_path, 'w') as f:
                json.dump(content, f, indent=2)

    params = files['params.json'][2]
    material_ids = set(params['segment']['materials'].values())

    with session_scope() as sess:
        rendering = models.Rendering(
            dataset_name=dataset_name,
            client=client_id,
            epoch=epoch,
            split_set=models.SplitSet[split_set.upper()],
            pair_id=int(pair_id),
            index=int(iteration),
            prefix=prefix,
            saturated_frac=saturated_frac,
            rend_time=params['time_elapsed'],
        )
        for material_id in material_ids:
            material = request.app['material_by_id'][material_id]
            rendering.materials.append(material)
        sess.add(rendering)
        sess.commit()
        logger.info('registered',
                    rendering_id=rendering.id,
                    dataset=rendering.dataset_name,
                    client=rendering.client,
                    epoch=rendering.epoch,
                    split_set=rendering.split_set,
                    pair_id=rendering.pair_id)

    return web.json_response(data={'status': 'success'})


def main():
    app = web.Application(middlewares=[
        normalize_path_middleware(),
    ])
    app.router.add_route('GET', '/', hello, name='hello')
    app.router.add_route('POST', '/submit', data_handler,
                         name='data_handler')

    with session_scope() as sess:
        materials = sess.query(models.Material).all()
        app['material_by_id'] = {m.id: m for m in materials}

    web.run_app(app, host=args.host, port=args.port, print=logger.info)


if __name__ == '__main__':
    main()
