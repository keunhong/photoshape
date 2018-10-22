import click
from tqdm import tqdm

from rendkit.graphics_utils import compute_tight_clipping_planes
from rendkit.shortcuts import render_segments
from terial import config
from terial.database import session_scope
from terial.models import ExemplarShapePair
from toolbox import cameras
from toolbox.images import mask_bbox

from vispy import app

app.use_app('glfw')


@click.command()
def main():
    with session_scope() as sess:
        pairs = (sess.query(ExemplarShapePair)
                 .filter(ExemplarShapePair.distance < config.ALIGN_DIST_THRES)
                 .all())

        print(f'Fetched {len(pairs)} pairs. '
              f'align_dist_thres = {config.ALIGN_DIST_THRES}',
              len(pairs), config.ALIGN_DIST_THRES)

        pbar = tqdm(pairs)
        for pair in pbar:
            pbar.set_description(f'[Pair {pair.id}]')
            mesh, materials = pair.shape.load()
            camera = cameras.spherical_coord_to_cam(
                pair.fov, pair.azimuth, pair.elevation)
            camera.near, camera.far = \
                compute_tight_clipping_planes(mesh, camera.view_mat())
            segment_im = render_segments(mesh, camera)
            fg_bbox = mask_bbox(segment_im > -1)
            pair.params = {
                'camera': camera.tojsd(),
                'crop_bbox': fg_bbox,
            }
            sess.commit()


if __name__ == '__main__':
    main()
