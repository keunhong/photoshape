import aiohttp
from tempfile import NamedTemporaryFile

import json

from terial.classifier.data.test_collector import (array_to_png_file,
                                                   array_to_jpg_file)
from toolbox.io.images import save_hdr


async def send_data(
        sess, client_id, split_set, pair_id, epoch, iteration,
        ldr_image, hdr_image, seg_map, seg_vis, normal_image,
        params, *,
        host, port, protocol='https', login='terial',
        password='NRXdKMTypAJGEENPcFmDCeybLPpd2KBfGUWFcopq8JBgqZoRnaVpyBpnNWdrJkHV'):

    metadata = {
        'split_set': split_set,
        'client_id': client_id,
        'pair_id': pair_id,
        'epoch': epoch,
        'iteration': iteration,
    }

    auth = aiohttp.BasicAuth(login=login, password=password)

    with NamedTemporaryFile(suffix='.exr') as hdr_f:
        save_hdr(hdr_f.name, hdr_image)
        data = aiohttp.FormData()
        data.add_field('metadata',
                       json.dumps(metadata),
                       filename='metadata.json',
                       content_type='application/json')
        data.add_field('params',
                       json.dumps(params),
                       filename='params.json',
                       content_type='application/json')
        data.add_field('ldr', array_to_jpg_file(ldr_image),
                       filename='ldr.jpg', content_type='image/jpeg')
        data.add_field('hdr', open(hdr_f.name, 'rb'),
                       filename='hdr.exr', content_type='image/x-exr')
        data.add_field('segment_map', array_to_png_file(seg_map),
                       filename='segment_map.png', content_type='image/png')
        data.add_field('segment_vis', array_to_png_file(seg_vis),
                       filename='segment_vis.png', content_type='image/png')
        data.add_field('normal_image', array_to_png_file(normal_image),
                       filename='normal_image.png', content_type='image/png')

        async with sess.post(f'{protocol}://{host}:{port}/submit',
                             data=data, auth=auth) as r:
            print(await r.json())
