import io

import aiohttp
import asyncio
import click

import numpy as np
from PIL import Image

from terial.classifier.data import collector


def array_to_png_file(array):
    if len(array.shape) == 2:
        mode = 'L'
    elif len(array.shape) == 3:
        if array.shape[2] == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'
    else:
        raise RuntimeError()

    image = Image.fromarray(array, mode=mode)
    output = io.BytesIO()
    image.save(output, format='png')
    return output.getbuffer()


def array_to_jpg_file(array):
    if len(array.shape) == 2:
        mode = 'L'
    elif len(array.shape) == 3:
        if array.shape[2] == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'
    else:
        raise RuntimeError()

    image = Image.fromarray(array, mode=mode)
    output = io.BytesIO()
    image.save(output, format='jpeg', quality=90)
    return output.getbuffer()


async def send_request(host, port):
    http_sess = aiohttp.ClientSession()

    ldr_image = np.ones((500, 500, 3), dtype=np.uint8) * 128
    hdr_image = np.zeros((500, 500, 3), dtype=np.float32)
    segment_map_image = np.zeros((500, 500))
    segment_vis_image = np.zeros((500, 500, 3))

    await collector.send_data(
        http_sess,
        client_id='test-client-1',
        pair_id=111,
        epoch=1,
        iteration=2,
        ldr_image=ldr_image,
        hdr_image=hdr_image,
        seg_map=segment_map_image,
        seg_vis=segment_vis_image,
        host=host,
        port=port)



@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=9500)
def main(host, port):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_request(host, port))


if __name__ == '__main__':
    main()
