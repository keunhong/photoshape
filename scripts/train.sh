#!/bin/sh

today=$(date +%Y%m%d.%H%M%S)

LMDB_NAME=20180503-nosat-384x384
SNAPSHOT_DIR=/local1/kpar/data/terial/brdf-classifier/lmdb/$LMDB_NAME/
CHECKPOINT_DIR=/projects/grail/kparnb/data/terial/brdf-classifier/checkpoints/$LMDB_NAME/

BASE_MODEL=resnet18

hostname=$(hostname -s)
INIT_LR=0.001
MAX_EPOCHS=100
START_EPOCH=0
LR_DECAY_EPOCHS=100
LR_DECAY_FRAC=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.0004
BATCH_SIZE=240
MASK_NOISE_P=0.0

SUBSTANCE_LOSS=fc
COLOR_LOSS=none

COLOR_HIST_NAME=none

MATERIAL_VARIANCE_INIT=0.0
SUBSTANCE_VARIANCE_INIT=-0.7
COLOR_VARIANCE_INIT=0.0

SUBSTANCE_LOSS_WEIGHT=0.5
COLOR_LOSS_WEIGHT=0.5

VISDOM_PORT=12345
NUM_WORKERS=12
SHOW_FREQ=10

MODEL_NAME="$today.$BASE_MODEL.subst_loss=$SUBSTANCE_LOSS.color_loss=$COLOR_LOSS.$COLOR_HIST_NAME.lr=$INIT_LR.mask_noise_p=$MASK_NOISE_P.use_variance.$hostname"

command="python -m terial.classifier.masked_input.train \
  --snapshot-dir $SNAPSHOT_DIR \
  --checkpoint-dir $CHECKPOINT_DIR \
  --base-model $BASE_MODEL \
  --model-name $MODEL_NAME \
  --init-lr $INIT_LR \
  --epochs $MAX_EPOCHS \
  --start-epoch $START_EPOCH \
  --lr-decay-epochs $LR_DECAY_EPOCHS \
  --lr-decay-frac $LR_DECAY_FRAC \
  --batch-size $BATCH_SIZE \
  --momentum $MOMENTUM \
  --weight-decay $WEIGHT_DECAY \
  --substance-loss $SUBSTANCE_LOSS \
  --substance-loss-weight $SUBSTANCE_LOSS_WEIGHT \
  --color-loss $COLOR_LOSS \
  --color-loss-weight $COLOR_LOSS_WEIGHT \
  --color-hist-name $COLOR_HIST_NAME \
  --material-variance-init $MATERIAL_VARIANCE_INIT \
  --substance-variance-init $SUBSTANCE_VARIANCE_INIT \
  --color-variance-init $COLOR_VARIANCE_INIT \
  --visdom-port $VISDOM_PORT \
  --num-workers $NUM_WORKERS \
  --show-freq $SHOW_FREQ \
  --mask-noise-p $MASK_NOISE_P \
  --use-variance
"

echo $command
