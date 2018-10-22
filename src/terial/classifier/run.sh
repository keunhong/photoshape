#!/bin/sh

SNAPSHOT_NAME="snapshot-20180308-shape-envmap-split"

python -m terial.brdf_classifier.train \
  --train-path /local1/data/terial/brdf-classifier/snapshots/$SNAPSHOT_NAME/examples_train.json \
  --validation-path /local1/data/terial/brdf-classifier/snapshots/$SNAPSHOT_NAME/examples_validation.json \
	--checkpoint-dir /local1/data/terial/brdf-classifier/snapshots/$SNAPSHOT_NAME/checkpoints \
	--batch-size 128 \
	--model-name resnet18-envmap-split \
	--lr 0.0001

