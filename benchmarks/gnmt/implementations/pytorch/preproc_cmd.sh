#!/bin/bash
set -x

export PATH=/mnt/driver/bin:$PATH
export LD_LIBRARY_PATH=/mnt/driver/lib64:$LD_LIBRARY_PATH

MAX_SEQ_LEN=${MAX_SEQ_LEN:-75}
DATASET_DIR='/data'
PREPROC_DATADIR='/preproc_data'

cd /mnt/current
python preprocess_data.py \
    --preproc-data-dir ${PREPROC_DATADIR} \
    --dataset-dir ${DATASET_DIR} \
    --max-length-train ${MAX_SEQ_LEN}
