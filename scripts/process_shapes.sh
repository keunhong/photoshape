#!/bin/bash

export DISPLAY=:1

python -m terial.shapes.preprocess_shapenet --category $1
python -m terial.shapes.register_shapenet_csv /projects/grail/kparnb/data/shapenet/shapenet.$1.csv --category $1
python -m terial.shapes.generate_phong_renderings
python -m terial.shapes.generate_alignment_rends
python -m terial.shapes.generate_alignment_features
