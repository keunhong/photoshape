from typing import Tuple, Iterable

import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy.orm import joinedload

from terial import config, models
from terial.models import (ExemplarShapePair, Exemplar, Material, Shape,
                           ResultAnnotation)
from toolbox import logging


logger = logging.init_logger(__name__)


def fetch_exemplars(
        sess,
        page_size=None,
        page=0,
        order_by=Exemplar.id.asc(),
        skip_excluded=True,
        filters=None) -> Tuple[Iterable[Exemplar], int]:
    if filters is None:
        filters = []
    else:
        filters = [f for f in filters]

    if skip_excluded:
        filters.append(ExemplarShapePair.shape.has(exclude=False))

    query = (sess.query(Exemplar)
             .order_by(order_by)
             .filter(sa.and_(*filters)))
    count = query.count()

    if page_size is not None:
        query = query.offset(page * page_size).limit(page_size)

    exemplars = query.all()
    return exemplars, count


def fetch_pairs(
        sess,
        by_shape=False,
        max_dist=None,
        page_size=None,
        page=0,
        filters=None,
        order_by=ExemplarShapePair.id.asc(),
        by_shape_topk=5,
        exclude_shapes=True,
        exclude_exemplars=True) -> Tuple[Iterable[ExemplarShapePair], int]:
    if filters is None:
        filters = []
    else:
        filters = [f for f in filters]

    if by_shape:
        filters.append(ExemplarShapePair.rank <= by_shape_topk)

    if max_dist:
        filters.append(ExemplarShapePair.distance <= max_dist)

    if exclude_shapes:
        filters.append(ExemplarShapePair.shape.has(exclude=False))

    if exclude_exemplars:
        filters.append(ExemplarShapePair.exemplar.has(exclude=False))

    query = (sess.query(ExemplarShapePair)
             .options(joinedload(ExemplarShapePair.exemplar),
                      joinedload(ExemplarShapePair.shape))
             .order_by(order_by, ExemplarShapePair.distance.asc())
             .filter(sa.and_(*filters)))

    count = query.count()

    if page_size is not None:
        query = query.offset(page * page_size).limit(page_size)

    pairs = query.all()

    return pairs, count


def fetch_pairs_default(sess, page=0, page_size=None, filters=None):
    if filters is None:
        filters = []

    filters.extend([
        # ExemplarShapePair.num_segments >= ExemplarShapePair.num_substances,
        # ExemplarShapePair.num_segments.isnot(None),
    ])

    return fetch_pairs(
        sess,
        by_shape=True,
        by_shape_topk=config.INFERENCE_TOPK,
        max_dist=config.INFERENCE_MAX_DIST,
        order_by=ExemplarShapePair.shape_id.asc(),
        filters=filters,
        page=page,
        page_size=page_size,
    )


def fetch_shapes_with_annotations(sess, result_set_id, shape_source=None,
                                  filters=None):
    if filters is None:
        filters = []
    if shape_source and shape_source != 'all':
        filters.append(Shape.source == shape_source)

    query = (sess.query(Shape)
             .join(Shape.pairs)
             .outerjoin(Shape.result_annotations)
             .filter(sa.and_(*filters))
             .group_by(Shape.id)
             .having(func.count(ResultAnnotation.id) > 0))
    return query.all()


def fetch_pairs_with_annotations(sess, result_set_id, shape_source=None,
                                 filters=None):
    if filters is None:
        filters = []
    if shape_source and shape_source != 'all':
        filters.append(Shape.source == shape_source)

    query = (sess.query(ExemplarShapePair)
             .join(ExemplarShapePair.shape)
             .outerjoin(ExemplarShapePair.result_annotations)
             .filter(sa.and_(*filters))
             .group_by(ExemplarShapePair.id)
             .having(func.count(ResultAnnotation.id) > 0))
    return query.all()


def fetch_random_shape_with_pair(sess,
                                 username,
                                 filters=None,
                                 shape_source='all',
                                 max_dist=config.INFERENCE_MAX_DIST,
                                 include_annotated=False):
    if filters is None:
        filters = []
    else:
        filters = [f for f in filters]

    logger.info('!!! %s', shape_source)
    if shape_source is not None and shape_source != 'all':
        filters.append(Shape.source == shape_source)

    filters.extend([
        Shape.pairs.any(ExemplarShapePair.distance < max_dist),
        Shape.exclude == False,
        ])

    # Try to find un-annotated shapes first
    query = (sess.query(Shape)
             .join(Shape.pairs)
             .outerjoin(Shape.result_annotations)
             .filter(sa.and_(*filters))
             .group_by(Shape.id)
             .having(func.count(ResultAnnotation.id) == 0))
    logger.info(query.count())
    if query.count() == 0:
        query = (sess.query(Shape)
            .join(Shape.result_annotations)
            .filter(sa.and_(*filters,
                            ResultAnnotation.username != username)))

    query = query.order_by(func.random())
    return query.first()


def fetch_shapes_with_pairs(sess, page=0, page_size=None,
                            max_dist=config.INFERENCE_MAX_DIST,
                            filters=None):
    if filters is None:
        filters = []
    else:
        filters = [f for f in filters]

    filters.extend([
        Shape.pairs.any(ExemplarShapePair.distance < max_dist),
        Shape.exclude ==False,
    ])

    pair_query = (sess.query(models.Shape)
                  .outerjoin(models.Shape.pairs)
                  .filter(sa.and_(*filters))
                  .order_by(Shape.id))
    pair_count = pair_query.count()

    query = (sess.query(models.Shape)
             .join(models.Shape.pairs)
             .filter(sa.and_(*filters))
             .order_by(Shape.id)
             .group_by(Shape.id))
    count = query.count()
    if page_size is not None:
        query = query.offset(page * page_size).limit(page_size)

    return query.all(), count, pair_count


def fetch_materials(
        sess,
        page_size=None,
        page=0,
        order_by=Material.id.asc(),
        include_disabled=True,
        filters=None) -> Tuple[Iterable[Material], int]:
    if filters is None:
        filters = []
    else:
        filters = [f for f in filters]

    if not include_disabled:
        filters.append(Material.enabled.is_(True))

    if not isinstance(order_by, tuple):
        order_by = (order_by,)

    query = (sess.query(Material)
             .order_by(*order_by)
             .filter(sa.and_(*filters)))
    count = query.count()

    if page_size is not None:
        query = query.offset(page * page_size).limit(page_size)

    materials = query.all()
    return materials, count
