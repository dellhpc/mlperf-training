#/bin/bash

# DL params
#SEED=125904
MAX_TOKENS=8192
LEARNING_RATE="1.976e-3"
WARMUP_UPDATES=1000
EXTRA_PARAMS="--max-source-positions 80 --max-target-positions 80 --enable-parallel-backward-allred-opt --parallel-backward-allred-opt-threshold 147566182 --parallel-backward-allred-cuda-nstreams 2 --adam-betas (0.9,0.98) "


# hardware params
SOCKETCORES=20
NSOCKET=2
