import argparse

from pathlib import Path

from tqdm import tqdm
import sqlalchemy as sa

from terial import models
from terial.database import session_scope
from terial.models import SplitSet

parser = argparse.ArgumentParser()
parser.add_argument(dest='dataset_dir', type=Path)
args = parser.parse_args()


SHAPE = (500, 500)


def main():
    tqdm.write(f"Fetching renderings from DB.")
    with session_scope() as sess:
        rends = (
            sess.query(models.Rendering)
                .all())

        num_excluded = 0

        pbar = tqdm(rends)
        for rend in pbar:
            if (rend.pair.exclude
                    or rend.pair.shape.exclude
                    or rend.pair.exemplar.exclude):
                pbar.set_description(f"{rend.id} is excluded")
                num_excluded += 1
                rend.exclude = True
            else:
                pbar.set_description(f"{rend.id} is not excluded")

        print(f"A total of {num_excluded} renderings were marked as excluded")
        if input('Commit to db? (y/n)') == 'y':
            sess.commit()


if __name__ == '__main__':
    main()
