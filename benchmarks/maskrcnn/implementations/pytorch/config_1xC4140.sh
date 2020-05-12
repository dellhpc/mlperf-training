#!/bin/bash

# DL params
EXTRA_PARAMS=""
EXTRA_CONFIG=(
	       "SOLVER.BASE_LR"       "0.03"
	       "SOLVER.MAX_ITER"      "160"
	       "SOLVER.WARMUP_FACTOR" "0.000048"
	       "SOLVER.WARMUP_ITERS"  "625"
	       "SOLVER.WARMUP_METHOD" "mlperf_linear"
	       "SOLVER.STEPS"         "(48000, 64000)"
	       "SOLVER.IMS_PER_BATCH"  "24"
	       "TEST.IMS_PER_BATCH" "4"
	       "MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN" "6000"
	       "NHWC" "True"   
	     )

#	       "SOLVER.MAX_ITER"      "160000"

# hardware config params
SOCKETCORES=20
NSOCKET=2
