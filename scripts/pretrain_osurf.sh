today=$(date +%Y-%m-%d.%H-%M-%S)

BASE_MODEL=resnet34
CHECKPOINT_DIR=/local1/kpar/data/terial/brdf-classifier/pretrain/

hostname=$(hostname -s)
INIT_LR=0.0001
LR_DECAY_EPOCHS=30
LR_DECAY_FRAC=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.0004
BATCH_SIZE=160

SUBSTANCE_LOSS=fc
SUBSTANCE_LOSS_WEIGHT=0.5

COLOR_LOSS=cross_entropy
COLOR_LOSS_WEIGHT=0.5
COLOR_HIST_NAME=base_color_hist_lab_3_5_5

VISDOM_PORT=8097
NUM_WORKERS=4
SHOW_FREQ=10

MODEL_NAME="pretrain.$today.$hostname.$BASE_MODEL.opensurfaces.subst_fc.color_lab_3_5_5"

command="python -m terial.classifier.opensurfaces.pretrain \
  --opensurfaces-dir /local1/kpar/data/opensurfaces \
  --checkpoint-dir $CHECKPOINT_DIR \
  --color-dir /local1/kpar/data/opensurfaces/shapes-colorhists/base_color_hist_lab_3_5_5 \
  --base-model $BASE_MODEL \
  --model-name $MODEL_NAME \
  --init-lr $INIT_LR \
  --lr-decay-epochs $LR_DECAY_EPOCHS \
  --lr-decay-frac $LR_DECAY_FRAC \
  --batch-size $BATCH_SIZE \
  --momentum $MOMENTUM \
  --weight-decay $WEIGHT_DECAY \
  --substance-loss $SUBSTANCE_LOSS \
  --color-loss $COLOR_LOSS \
  --visdom-port $VISDOM_PORT \
  --num-workers $NUM_WORKERS \
  --show-freq $SHOW_FREQ \
  --use-variance
"

echo $command
exec $command
