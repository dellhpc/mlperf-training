#!/bin/bash

sandbox="sandbox-pytorch-ngc19.05"
singularity pull docker://nvcr.io/nvidia/pytorch:19.05-py3
singularity build --sandbox $sandbox pytorch_19.05-py3.sif
singularity exec -w $sandbox mkdir -p /mnt/current /mnt/driver /data
singularity exec -w -B $PWD:/mnt/current $sandbox pip install --no-cache-dir https://github.com/mlperf/training/archive/6289993e1e9f0f5c4534336df83ff199bd0cdb75.zip#subdirectory=compliance
singularity exec -w -B $PWD:/mnt/current $sandbox pip install --no-cache-dir -r /mnt/current/requirements.txt
singularity exec -w -B /cm/local/apps/cuda/libs/current:/mnt/driver -B $PWD:/mnt/current $sandbox bash /mnt/current/build_ssd.sh
