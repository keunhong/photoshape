import argparse
import time
from pathlib import Path

import ujson as json
import skimage.io

import numpy as np
import visdom
from tqdm import tqdm

from terial.classifier.data import RawDataset, utils
from toolbox.images import apply_mask

parser = argparse.ArgumentParser()
parser.add_argument(dest='dataset_dir', type=Path)
parser.add_argument('--visdom-port', default=8097)
args = parser.parse_args()


vis = visdom.Visdom(env='classifier-data-find-saturated',
                    port=args.visdom_port)


def main():
    dataset = RawDataset(args.dataset_dir, 'validation')

    num_over_thres = 0
    pbar = tqdm(dataset, total=len(dataset))
    for data_dir, prefix in pbar:
        params_path = data_dir / f'{prefix}.params.json'
        with params_path.open('r') as f:
            params_dict = json.load(f)

        if 'saturated_frac' in params_dict:
            continue

        seg_map_path = data_dir / f'{prefix}.segment_map.384x384.png'
        ldr_path = data_dir / f'{prefix}.ldr.384x384.png'
        seg_map = skimage.io.imread(seg_map_path)
        ldr_im = skimage.io.imread(ldr_path)
        fg_mask = seg_map > 0
        saturated_frac = utils.compute_saturated_frac(ldr_im, fg_mask)
        params_dict['saturated_frac'] = saturated_frac

        with params_path.open('w') as f:
            json.dump(params_dict, f)


        if saturated_frac > 0.05:
            num_over_thres += 1
            vis.text('<pre>'
                     f'num_over_thres = {num_over_thres}\n'
                     f'saturated_frac = {saturated_frac}\n'
                     '</pre>', win='info')
            vis.image(ldr_im.transpose((2, 0, 1)), win='image')

        pbar.set_description(f'{num_over_thres}, {saturated_frac:.02f}')

if __name__ == '__main__':
    main()
