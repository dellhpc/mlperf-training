#!/bin/bash

export PATH=/mnt/driver/bin:$PATH
export LD_LIBRARY_PATH=/mnt/driver/lib64:$LD_LIBRARY_PATH

cd /mnt/current
pip install --no-cache-dir https://github.com/mlperf/training/archive/6289993e1e9f0f5c4534336df83ff199bd0cdb75.zip#subdirectory=compliance
pip install --no-cache-dir -r /mnt/current/requirements.txt
python setup.py install
