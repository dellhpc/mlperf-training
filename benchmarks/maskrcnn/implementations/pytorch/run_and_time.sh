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

python -c "import mlperf_compliance; from maskrcnn_benchmark.utils.mlperf_logger import mlperf_submission_log; mlperf_submission_log(mlperf_compliance.constants.MASKRCNN)"

echo "Clearing caches."
sync && echo 3 | tee /proc/sys/vm/drop_caches
STAT=$?
if [ $STAT -eq 0 ]; then
    echo "Cache clear successful."
    python -c "import mlperf_compliance as mc;mc.mlperf_log.mlperf_print(mlperf_compliance.constants.CACHE_CLEAR, value=True, stack_offset=0)"
fi


cd /mnt/current    
source config_${SYSTEM}.sh

#[ ! -f /coco ] && ln -sf ${DATADIR} /coco
export SHARED_DIR=/mnt/shared
TMPFILE=$(mktemp --tmpdir=$SHARED_DIR)
touch $TMPFILE || exit 1
rm -f $TMPFILE

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt" 

#export NCCL_IB_HCA=^mlx5_1:1
export NCCL_DEBUG=info

# run training
#python3.6 -m torch.distributed.launch \
#    --nproc_per_node $gpus_per_node $MULTI_NODE \
python -m bind_launch \
  --no_hyperthreads \
  --nsockets_per_node ${NSOCKET} \
  --ncores_per_socket ${SOCKETCORES} \
  --nproc_per_node $gpus_per_node $MULTI_NODE train_mlperf.py \
  ${EXTRA_PARAMS} \
  --config-file 'configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
  DTYPE 'float16' \
  PATHS_CATALOG 'maskrcnn_benchmark/config/paths_catalog_dbcluster.py' \
  MODEL.WEIGHT '/coco/models/R-50.pkl' \
  DISABLE_REDUCED_LOGGING True \
  "${EXTRA_CONFIG[@]}" 

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt" 

#unlink /coco

# report result
result=$(( $end - $start ))
result_name="OBJECT_DETECTION"
echo "RESULT,${result_name},$SEED,$result,DELLEMC,$start_fmt" 

hours=$(( ${result}/3600 ))
minutes=$(( (${result}%3600)/60 ))
seconds=$(( ${result}%60 ))
total_mins=$(awk "BEGIN {printf \"%.2f\", $result/60}")
echo "The application took $result seconds, or $total_mins min, or ${hours} h:${minutes} m:${seconds} s"
