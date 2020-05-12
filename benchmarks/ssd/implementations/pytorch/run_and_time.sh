#!/bin/bash
set -x

SYSTEM=$1
gpus_per_node=$2
node_id=$3

set +x
export PATH=/mnt/driver/bin:$PATH
export LD_LIBRARY_PATH=/mnt/driver/lib64:$LD_LIBRARY_PATH
set -x

# Configure environment variables
export OMP_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

if [[ $node_id -eq 0 ]]; then
    lscpu
    free -g
    nvidia-smi
    nvidia-smi topo -m
fi

SEED=${SEED:-$RANDOM}
echo "node $node_id has seed $SEED "

python -c "import mlperf_compliance; from mlperf_log_utils import mlperf_submission_log; mlperf_submission_log(mlperf_compliance.constants.SSD)"

echo "Clearing caches."
sync && echo 3 | tee /proc/sys/vm/drop_caches
STAT=$?
if [ $STAT -eq 0 ]; then
    echo "Cache clear successful."
    python -c "import mlperf_compliance as mc; mc.mlperf_log.mlperf_print(mc.constants.CACHE_CLEAR, value=True, stack_offset=0)"
fi


cd /mnt/current    
source config_${SYSTEM}.sh
export TORCH_MODEL_ZOO=/data/torchvision

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt" 

export NCCL_DEBUG=info
#export NCCL_IB_HCA=^mlx5_1:1

# run training
python bind_launch.py --nsockets_per_node ${NSOCKET} \
    --ncores_per_socket ${SOCKETCORES} \
    --no_hyperthreads \
    --nproc_per_node $gpus_per_node \
    train.py \
    --use-fp16 \
    --nhwc \
    --pad-input \
    --jit \
    --delay-allreduce \
    --opt-loss \
    --epochs $epochs \
    --warmup-factor 0 \
    --no-save \
    --threshold=0.23 \
    --data /data \
    --evaluation 120000 160000 180000 200000 220000 240000 260000 280000 \
    ${EXTRA_PARAMS[@]} 

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt" 

# report result
result=$(( $end - $start ))
result_name="OBJECT_DETECTION"
echo "RESULT,${result_name},$SEED,$result,DELLEMC,$start_fmt" 

hours=$(( ${result}/3600 ))
minutes=$(( (${result}%3600)/60 ))
seconds=$(( ${result}%60 ))
total_mins=$(awk "BEGIN {printf \"%.2f\", $result/60}")
echo "The application took $result seconds, or $total_mins min, or ${hours} h:${minutes} m:${seconds} s"
