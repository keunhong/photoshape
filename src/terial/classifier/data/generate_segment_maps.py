import ujson as json
from pathlib import Path

import numpy as np
import click
import tqdm
import visdom
from skimage.io import imsave

from rendkit import shortcuts
from terial import models
from terial.classifier import rendering_dataset
from terial.database import session_scope
from toolbox import cameras
import vispy.app
from toolbox.images import visualize_map

vispy.app.use_app('glfw')

vis = visdom.Visdom(env='generate-segment-maps')


@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True)
def main(data_dir):
    path_dicts = rendering_dataset.get_path_dicts(data_dir)

    pbar = tqdm.tqdm(path_dicts)
    for d in pbar:
        if Path(d['seg_map_path']).exists():
            continue

        with open(d['json_path'], 'r') as f:
            params = json.load(f)

        render_segment_map(pbar, d, params)


def render_segment_map(pbar, d, params):
    with session_scope() as sess:
        pair = sess.query(models.ExemplarShapePair).get(params['pair_id'])
        rk_mesh, _ = pair.shape.load()

    shape_id = params['shape_id']
    cam_fov = params['camera']['fov']
    cam_azimuth = params['camera']['azimuth']
    cam_elevation = params['camera']['elevation']
    cam_dist = params['camera']['distance']

    rk_mesh.resize(1)

    # Set random camera.
    rk_camera = cameras.spherical_coord_to_cam(
        cam_fov, cam_azimuth, cam_elevation, cam_dist=cam_dist, max_len=500)

    seg_map = shortcuts.render_segments(rk_mesh, rk_camera)
    seg_vis = visualize_map(seg_map)[:, :, :3]

    path = d['seg_vis_path']
    pbar.set_description(f'{path}')

    imsave(d['seg_vis_path'], seg_vis)
    imsave(d['seg_map_path'], (seg_map + 1).astype(np.uint8))

    vis.image(visualize_map(seg_map).transpose((2, 0, 1)))



if __name__ == '__main__':
    main()

