#!/bin/bash

# DL params
EXTRA_PARAMS=(
	       --batch-size      "32"
	       --eval-batch-size "160"
	       --warmup          "650"
	       --lr              "3.2e-3"
	       --wd              "1.3e-4"
	       --num-workers     "3"
	     )


# hardware config params
SOCKETCORES=20
NSOCKET=2
