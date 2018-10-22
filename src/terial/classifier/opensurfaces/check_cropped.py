from pathlib import Path

import skimage.io
from tqdm import tqdm


def main():
    cropped_dir = Path('/local1/kpar/data/opensurfaces/shapes-cropped')
    n_bad = 0
    pbar = tqdm(list(cropped_dir.iterdir()))
    for cropped_path in pbar:
        try:
            image = skimage.io.imread(str(cropped_path))
        except (OSError, ValueError) as e:
            tqdm.write(str(e))
            # cropped_path.unlink()
            n_bad += 1
            pbar.set_description(str(n_bad))


if __name__ == '__main__':
    main()
