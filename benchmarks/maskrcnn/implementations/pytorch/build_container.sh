#!/bin/bash

sandbox="sandbox-pytorch-ngc19.05"
singularity pull pytorch_19.05-py3.sif docker://nvcr.io/nvidia/pytorch:19.05-py3
singularity build --sandbox $sandbox pytorch_19.05-py3.sif
singularity exec -w $sandbox mkdir -p /mnt/current /mnt/driver /mnt/shared /coco
singularity exec -w -B /cm/local/apps/cuda/libs/current:/mnt/driver -B $PWD:/mnt/current $sandbox bash /mnt/current/build_maskrcnn.sh
