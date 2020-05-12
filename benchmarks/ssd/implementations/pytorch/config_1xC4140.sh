#!/bin/bash

# DL params
epochs=80
EXTRA_PARAMS=(
	       --batch-size      "120"
	       --eval-batch-size "160"
	       --warmup          "650"
	       --lr              "2.92e-3"
	       --wd              "1.6e-4"
	       --use-nvjpeg
	       --use-roi-decode
	     )


# hardware config params
SOCKETCORES=20
NSOCKET=2
