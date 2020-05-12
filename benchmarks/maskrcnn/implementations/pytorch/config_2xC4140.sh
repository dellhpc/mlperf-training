#!/bin/bash

# DL params
EXTRA_PARAMS=""
EXTRA_CONFIG=(
	       "SOLVER.BASE_LR"       "0.06"
	       "SOLVER.MAX_ITER"      "800"
	       "SOLVER.WARMUP_FACTOR" "0.000096"
	       "SOLVER.WARMUP_ITERS"  "625"
	       "SOLVER.WARMUP_METHOD" "mlperf_linear"
	       "SOLVER.STEPS"         "(24000, 32000)"
	       "SOLVER.IMS_PER_BATCH"  "48"
	       "TEST.IMS_PER_BATCH" "8"
	       "MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN" "6000"
	       "NHWC" "True"   
	     )
#	       "SOLVER.MAX_ITER"      "80000"

# hardware config params
SOCKETCORES=20
NSOCKET=2
