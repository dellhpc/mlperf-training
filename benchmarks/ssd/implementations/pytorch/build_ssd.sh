#!/bin/bash

export PATH=/mnt/driver/bin:$PATH
export LD_LIBRARY_PATH=/mnt/driver/lib64:$LD_LIBRARY_PATH

cd /mnt/current
python setup.py install
