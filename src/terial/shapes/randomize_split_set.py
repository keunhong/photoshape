"""
Assigns the split set for each shape to either 'train' or 'validation'.
"""
import random

from terial.database import session_scope
from terial.models import Shape


def main():
    with session_scope() as sess:
        shapes = list(sess.query(Shape).filter_by(exclude=False).all())
        random.shuffle(shapes)

        n_train = int(0.9 * len(shapes))
        for shape in shapes[:n_train]:
            shape.split_set = 'train'

        for shape in shapes[n_train:]:
            shape.split_set = 'validation'

        sess.commit()


if __name__ == '__main__':
    main()
