#!/bin/bash
set -x

SYSTEM=$1

# GPU driver path
set +x
export PATH=/mnt/driver/bin:$PATH
export LD_LIBRARY_PATH=/mnt/driver/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
set -x

if [[ $OMPI_COMM_WORLD_RANK -eq 0 ]]; then
    python -c "import mxnet as mx; print(mx.__version__)"
    hostname
    lscpu
    free -g
    nvidia-smi
    nvidia-smi topo -m
fi

# container wise env
export MXNET_UPDATE_ON_KVSTORE=0      
export MXNET_EXEC_ENABLE_ADDTO=1      
export MXNET_USE_TENSORRT=0           
export MXNET_GPU_WORKER_NTHREADS=1    
export MXNET_GPU_COPY_NTHREADS=1      
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0 
export MXNET_OPTIMIZER_AGGREGATION_SIZE=54 
export NCCL_BUFFSIZE=2097152          
export NCCL_NET_GDR_READ=1            
export HOROVOD_CYCLE_TIME=0.2         
export HOROVOD_BATCH_D2D_MEMCOPIES=1  
export HOROVOD_GROUPED_ALLREDUCES=1  
export HOROVOD_NUM_STREAMS=1  
export MXNET_HOROVOD_NUM_GROUPS=1 
export NCCL_MAX_NRINGS=8 
export OMP_NUM_THREADS=1 
export OPENCV_FOR_THREADS_NUM=1

SEED=${SEED:-$RANDOM}

echo "Clearing caches."
sync && echo 3 | tee /proc/sys/vm/drop_caches
STAT=$?
if [ $STAT -eq 0 ]; then
    echo "Cache clear successful."
fi


cd /mnt/current
source config_${SYSTEM}.sh

# the following command will cause MPI error, need to figure out why
#python -c "import mlperf_compliance; from mlperf_log_utils import mlperf_submission_log; mlperf_submission_log(mlperf_compliance.constants.RESNET)"

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt" 

OPTIMIZER=${OPTIMIZER:-"sgd"}
BATCHSIZE=${BATCHSIZE:-1664}
KVSTORE=${KVSTORE:-"device"}
LR=${LR:-"0.6"}
LRSCHED=${LRSCHED:-"30,60,80"}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
LARSETA=${LARSETA:-'0.001'}
WD=${WD:-'0.0001'}
LABELSMOOTHING=${LABELSMOOTHING:-'0.0'}
SEED=${SEED:-1}
EVAL_OFFSET=${EVAL_OFFSET:-2}
EVAL_PERIOD=${EVAL_PERIOD:-4}
DALI_PREFETCH_QUEUE=${DALI_PREFETCH_QUEUE:-2}
DALI_NVJPEG_MEMPADDING=${DALI_NVJPEG_MEMPADDING:-64}
DALI_THREADS=${DALI_THREADS:-3}
DALI_CACHE_SIZE=${DALI_CACHE_SIZE:-0}
DALI_ROI_DECODE=${DALI_ROI_DECODE:-0}
NUMEPOCHS=${NUMEPOCHS:-90}
NETWORK=${NETWORK:-"resnet-v1b-fl"}

if [[ "$OPTIMIZER" == "sgdwlars" ]] || [[ "$OPTIMIZER" == "sgdwfastlars" ]]; then
    THR="0.759"
else
    THR="0.749"
fi

# run training
gpus_per_node=4
GPUS=$(seq 0 $(($gpus_per_node - 1)) | tr "\n" "," | sed 's/,$//')
PARAMS=(
      --gpus               "${GPUS}"
      --batch-size         "${BATCHSIZE}"
      --kv-store           "${KVSTORE}"
      --lr                 "${LR}"
      --lr-step-epochs     "${LRSCHED}"
      --lars-eta           "${LARSETA}"
      --label-smoothing    "${LABELSMOOTHING}"
      --wd                 "${WD}"
      --warmup-epochs      "${WARMUP_EPOCHS}"
      --eval-period        "${EVAL_PERIOD}"
      --eval-offset        "${EVAL_OFFSET}"
      --optimizer          "${OPTIMIZER}"
      --network            "${NETWORK}"
      --num-layers         "50"
      --num-epochs         "${NUMEPOCHS}"
      --accuracy-threshold "${THR}"
      --seed               "${SEED}"
      --dtype              "float16"
      --use-dali 
      --disp-batches       "20"
      --image-shape        "4,224,224"
      --fuse-bn-relu       "1"
      --fuse-bn-add-relu   "1"
      --min-random-area    "0.05"
      --max-random-area    "1.0"
      --conv-algo          "1"
      --force-tensor-core  "1"
      --input-layout       "NHWC"
      --conv-layout        "NHWC"
      --batchnorm-layout   "NHWC"
      --pooling-layout     "NHWC"
      --batchnorm-mom      "0.9"
      --batchnorm-eps      "1e-5"
      --data-train         "/data/train.rec"
      --data-train-idx     "/data/train.idx"
      --data-val           "/data/val.rec"
      --data-val-idx       "/data/val.idx"
      --dali-prefetch-queue        "${DALI_PREFETCH_QUEUE}"
      --dali-nvjpeg-memory-padding "${DALI_NVJPEG_MEMPADDING}"
      --dali-threads       "${DALI_THREADS}"
      --dali-cache-size    "${DALI_CACHE_SIZE}"
      --dali-roi-decode    "${DALI_ROI_DECODE}"
)

export NCCL_DEBUG=info
#export NCCL_IB_HCA=^mlx5_1:1

python train_imagenet.py "${PARAMS[@]}"

set +x
# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)

# report result
result=$(( $end - $start ))
result_name="IMAGE_CLASSIFICATION"

hours=$(( ${result}/3600 ))
minutes=$(( (${result}%3600)/60 ))
seconds=$(( ${result}%60 ))
total_mins=$(awk "BEGIN {printf \"%.2f\", $result/60}")
set -x

echo "ENDING TIMING RUN AT $end_fmt" 
echo "RESULT,${result_name},$SEED,$result,$USER,$start_fmt" 
echo "The application took $result seconds, or $total_mins min, or ${hours} h:${minutes} m:${seconds} s"
