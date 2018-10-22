BRDF Inference
==============

### Creating Renderings

In order to train our network we need to create a large number of random renderings. This is done with the command

```
python -m terial.brdf_classifier.tools.generate_data \
  --client-id {$CLIENT_ID} \
  --epoch-start 0 \
  --n-epochs 100 \
  --n-rends-per-epoch 10 \
  --out-dir {$RANDOM_REND_OUT_DIR}
```

### Creating Snapshot

Once we have enough data, we must create a training/validation split. We call this a 'snapshot' and it is created with
the following command

```
python -m terial.brdf_classifier.tools.make_data_snapshot \
  --train-frac 0.9 \
  --split-shapes \
  --split-envmaps \
  {$RANDOM_REND_OUT_DIR} {$SNAPSHOT_OUT_PATH}
```


### Training Network

Now we can train a network

```
python -m terial.brdf_classifier.train \
  --train-path {$SNAPSHOT_OUT_PATH}/examples_train.json \
  --validation-path {$SNAPSHOT_OUT_PATH}/examples_validation.json \
  --checkpoint-dir {$SNAPSHOT_OUT_PATH}/checkpoints \
  --model-name {$MODEL_NAME} \
  --batch-size 32 \
  --start-epoch 0 \
  --epochs 90 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 0.0001
```


