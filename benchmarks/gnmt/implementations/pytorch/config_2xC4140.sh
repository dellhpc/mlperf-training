#/bin/bash

# DL params
LR="2.0e-3"
TRAIN_BATCH_SIZE=256
TEST_BATCH_SIZE=128
WARMUP_STEPS=200
REMAIN_STEPS=6453
DECAY_INTERVAL=809
TARGET=24.0
MAX_SEQ_LEN=75
NUMEPOCHS=8
MATH=amp_fp16
EXTRA_OPTS="--fused-attention \
   	    --fused-xentropy \
            --no-log-all-ranks \
           "

# hardware params
SOCKETCORES=20
NSOCKET=2
