#!/bin/bash
set -x

SYSTEM=$1
gpus_per_node=$2
node_id=$3

set +x
export PATH=/mnt/driver/bin:$PATH
export LD_LIBRARY_PATH=/mnt/driver/lib64:$LD_LIBRARY_PATH
set -x

if [[ $node_id -eq 0 ]]; then
    lscpu
    free -g
    nvidia-smi
    nvidia-smi topo -m
fi

SEED=${SEED:-$RANDOM}
echo "node $node_id has seed $SEED "

python -c "import mlperf_compliance; from mlperf_log_utils import mlperf_submission_log; mlperf_submission_log(mlperf_compliance.constants.GNMT)"

echo "Clearing caches."
sync && echo 3 | tee /proc/sys/vm/drop_caches
STAT=$?
if [ $STAT -eq 0 ]; then
    echo "Cache clear successful."
    python -c "import mlperf_compliance as mc;mc.mlperf_log.mlperf_print(mc, value=True, stack_offset=0)"
fi

cd /mnt/current 
source config_${SYSTEM}.sh

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
SHARED_DIR='/mnt/shared'
DATASET_DIR='/data'
PREPROC_DATADIR='/preproc_data'

LR=${LR:-"2.0e-3"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-128}
WARMUP_STEPS=${WARMUP_STEPS:-200}
REMAIN_STEPS=${REMAIN_STEPS:-10336}
DECAY_INTERVAL=${DECAY_INTERVAL:-1296}
TARGET=${TARGET:-24.0}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-75}
NUMEPOCHS=${NUMEPOCHS:-8}
EXTRA_OPTS=${EXTRA_OPTS:-""}
BIND_LAUNCH=${BIND_LAUNCH:-0}
MATH=${MATH:-fp16}

if [[ $BIND_LAUNCH -eq 1 ]]; then
  LAUNCH_OPT="bind_launch  --nsockets_per_node ${NSOCKET}  --ncores_per_socket ${SOCKETCORES} --nproc_per_node ${gpus_per_node} ${MULTI_NODE}" 
else
  LAUNCH_OPT="torch.distributed.launch --nproc_per_node ${gpus_per_node} ${MULTI_NODE}"
fi

echo "running benchmark"

# run training
python -m ${LAUNCH_OPT}  train.py \
  --save ${SHARED_DIR} \
  --dataset-dir ${DATASET_DIR} \
  --preproc-data-dir ${PREPROC_DATADIR} \
  --target-bleu $TARGET \
  --epochs "${NUMEPOCHS}" \
  --math ${MATH} \
  --max-length-train ${MAX_SEQ_LEN} \
  --print-freq 10 \
  --train-batch-size $TRAIN_BATCH_SIZE \
  --test-batch-size $TEST_BATCH_SIZE \
  --optimizer FusedAdam \
  --lr $LR \
  --warmup-steps $WARMUP_STEPS \
  --remain-steps $REMAIN_STEPS \
  --decay-interval $DECAY_INTERVAL \
  $EXTRA_OPTS 

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="RNN_TRANSLATOR"

echo "RESULT,$result_name,,$result,DELLEMC,$start_fmt"
hours=$(( ${result}/3600 ))
minutes=$(( (${result}%3600)/60 ))
seconds=$(( ${result}%60 ))
total_mins=$(awk "BEGIN {printf \"%.2f\", $result/60}")
echo "The application took $result seconds, or $total_mins min, or ${hours} h:${minutes} m:${seconds} s"
