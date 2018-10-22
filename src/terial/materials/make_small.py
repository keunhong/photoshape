import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from terial import models
from terial.database import session_scope
from toolbox.images import resize
from toolbox.io.images import save_hdr, load_hdr

parser = argparse.ArgumentParser()
parser.add_argument(dest='in_dir', type=Path)
parser.add_argument(dest='out_dir', type=Path)
args = parser.parse_args()


def main():
    pbar = tqdm(list(args.in_dir.glob('**/*.mdl')))
    for mdl_file in pbar:
        rel_path = Path(str(mdl_file).replace(str(args.in_dir), '').lstrip('/'))
        pbar.set_description(f'{rel_path}')
        in_path = args.in_dir / rel_path
        out_path = args.out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(in_path, out_path)

    pbar = tqdm(list(args.in_dir.glob('**/*.png')) +
                list(args.in_dir.glob('**/*.jpg')))
    for png_file in pbar:
        rel_path = Path(str(png_file).replace(str(args.in_dir), '').lstrip('/'))
        pbar.set_description(f'{rel_path}')
        in_path = args.in_dir / rel_path
        out_path = args.out_dir / rel_path
        if out_path.exists():
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(str(in_path))
        image = image.resize((500, 500), resample=Image.LANCZOS)
        image.save(str(out_path))

    pbar = tqdm(list(args.in_dir.glob('**/*.exr')))
    for exr_file in pbar:
        rel_path = Path(str(exr_file).replace(str(args.in_dir), '').lstrip('/'))
        pbar.set_description(f'{rel_path}')
        in_path = args.in_dir / rel_path
        out_path = args.out_dir / rel_path
        if out_path.exists():
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        image = load_hdr(in_path)
        image = resize(image, (500, 500))
        save_hdr(out_path, image)



if __name__ == '__main__':
    main()




