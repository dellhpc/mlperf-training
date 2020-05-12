#!/bin/bash

set -e

SEED=$1

#python3 convert_utf8_to_fairseq_binary.py --data_dir /workspace/translation/examples/translation/wmt14_en_de
python3 convert_utf8_to_fairseq_binary.py --data_dir /mnt/isilon/DeepLearning/database/mlperf/v0.6/wmt14_en_de
