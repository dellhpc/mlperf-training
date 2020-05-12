#!/bin/bash

# DL params
EXTRA_PARAMS=""
EXTRA_CONFIG=(
	       "SOLVER.BASE_LR"       "0.16"
	       "SOLVER.MAX_ITER"      "40000"
	       "SOLVER.WARMUP_FACTOR" "0.000256"
	       "SOLVER.WARMUP_ITERS"  "625"
	       "SOLVER.WARMUP_METHOD" "mlperf_linear"
	       "SOLVER.STEPS"         "(9000, 12000)"
	       "SOLVER.IMS_PER_BATCH" "192"
	       "TEST.IMS_PER_BATCH" "32"
	       "MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN" "6000"
	       "NHWC" "True"   
	     )


# hardware config params
SOCKETCORES=20
NSOCKET=2
