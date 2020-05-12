#/bin/bash

# DL params
MAX_TOKENS=12800
LEARNING_RATE="1.976e-3"
WARMUP_UPDATES=1000
EXTRA_PARAMS="--max-source-positions 64 --max-target-positions 64 --enable-parallel-backward-allred-opt --parallel-backward-allred-opt-threshold 105404416 --parallel-backward-allred-cuda-nstreams 2 --adam-betas (0.9,0.98) "


# hardware params
SOCKETCORES=20
NSOCKET=2
