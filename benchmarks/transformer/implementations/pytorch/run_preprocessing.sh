#!/bin/bash

set -e

SEED=$1

#cd /workspace/translation

# TODO: Add SEED to process_data.py since this uses a random generator (future PR)
#export PYTHONPATH=/research/transformer/transformer:${PYTHONPATH}
# Add compliance to PYTHONPATH
# export PYTHONPATH=/mlperf/training/compliance:${PYTHONPATH}

mkdir -p /mnt/isilon/DeepLearning/database/mlperf/v0.6/wmt14_en_de
mkdir -p /mnt/isilon/DeepLearning/database/mlperf/v0.6/wmt14_en_de/utf8

cp reference_dictionary.ende.txt /mnt/isilon/DeepLearning/database/mlperf/v0.6/wmt14_en_de/dict.en.txt
cp reference_dictionary.ende.txt /mnt/isilon/DeepLearning/database/mlperf/v0.6/wmt14_en_de/dict.de.txt

sed -i "1s/^/\'<lua_index_compat>\'\n/" /mnt/isilon/DeepLearning/database/mlperf/v0.6/wmt14_en_de/dict.en.txt
sed -i "1s/^/\'<lua_index_compat>\'\n/" /mnt/isilon/DeepLearning/database/mlperf/v0.6/wmt14_en_de/dict.de.txt

# TODO: make code consistent to not look in two places (allows temporary hack above for preprocessing-vs-training)
cp reference_dictionary.ende.txt /mnt/isilon/DeepLearning/database/mlperf/v0.6/wmt14_en_de/utf8/dict.en.txt
cp reference_dictionary.ende.txt /mnt/isilon/DeepLearning/database/mlperf/v0.6/wmt14_en_de/utf8/dict.de.txt

#wget https://raw.githubusercontent.com/tensorflow/models/master/official/transformer/test_data/newstest2014.en -O /mnt/isilon/DeepLearning/database/mlperf/0.6/wmt14_en_de/newstest2014.en
#wget https://raw.githubusercontent.com/tensorflow/models/master/official/transformer/test_data/newstest2014.de -O /mnt/isilon/DeepLearning/database/mlperf/0.6/wmt14_en_de/newstest2014.de

python3 preprocess.py --raw_dir /mnt/isilon/DeepLearning/database/mlperf/v0.6/raw_data --data_dir /mnt/isilon/DeepLearning/database/mlperf/v0.6/wmt14_en_de

