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
    hostname
    lscpu
    free -g
    nvidia-smi
    nvidia-smi topo -m
fi

SEED=${SEED:-$RANDOM}
MODE=${MODE:-TRAIN}
#NUMEPOCHS=${NUMEPOCHS:-1}
NUMEPOCHS=${NUMEPOCHS:-30}
echo "node $node_id has seed $SEED "

python -c "import mlperf_compliance; from mlperf_log_utils import mlperf_submission_log; mlperf_submission_log(mlperf_compliance.constants.TRANSFORMER)"

echo "Clearing caches."
sync && echo 3 | tee /proc/sys/vm/drop_caches
STAT=$?
if [ $STAT -eq 0 ]; then
    echo "Cache clear successful."
    python -c "import mlperf_compliance as mc;mc.mlperf_log.mlperf_print(mlperf_compliance.constants.CACHE_CLEAR, value=True, stack_offset=0)"
fi

cd /mnt/current 
source config_${SYSTEM}.sh

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt" 

export NCCL_DEBUG=info
#export NCCL_IB_HCA=^mlx5_1:1
#export NCCL_TREE_THRESHOLD=4294967296 

# run training
#python -m torch.distributed.launch \
#    --nproc_per_node $gpus_per_node $MULTI_NODE \
#nvprof --profile-api-trace none -f -o profile/transformer_${total_gpus}_%p.nvprof --profile-child-processes \
python -m bind_launch --no_hyperthreads \
                      --nsockets_per_node ${NSOCKET} \
                      --ncores_per_socket ${SOCKETCORES} \
                      --nproc_per_node $gpus_per_node $MULTI_NODE \
    train.py /data \
    --seed ${SEED} \
    --arch transformer_wmt_en_de_big_t2t \
    --share-all-embeddings \
    --optimizer adam \
    --adam-betas '(0.9, 0.997)' \
    --adam-eps "1e-9" \
    --clip-norm "0.0" \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr "0.0" \
    --warmup-updates ${WARMUP_UPDATES} \
    --lr ${LEARNING_RATE} \
    --min-lr "0.0" \
    --dropout "0.1" \
    --weight-decay "0.0" \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing "0.1" \
    --max-tokens ${MAX_TOKENS} \
    --max-epoch ${NUMEPOCHS} \
    --target-bleu "25.0" \
    --ignore-case \
    --no-save \
    --update-freq 1 \
    --fp16 \
    --seq-len-multiple 2 \
    --softmax-type "fast_fill" \
    --source_lang en \
    --target_lang de \
    --bucket_growth_factor 1.035 \
    --batching_scheme "v0p5_better" \
    --batch_multiple_strategy "dynamic" \
    --fast-xentropy \
    --max-len-a 1 \
    --max-len-b 50 \
    --lenpen 0.6 \
    --distributed-init-method "env://" \
    ${EXTRA_PARAMS} 

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt" 

# report result
result=$(( $end - $start ))
result_name="TRANSFORMER"
echo "RESULT,${result_name},$SEED,$result,DELLEMC,$start_fmt" 

hours=$(( ${result}/3600 ))
minutes=$(( (${result}%3600)/60 ))
seconds=$(( ${result}%60 ))
#total_mins=$(echo "scale=2; $result/60" | bc -l)
total_mins=$(awk "BEGIN {printf \"%.2f\", $result/60}")
echo "The application took $result seconds, or $total_mins min, or ${hours} h:${minutes} m:${seconds} s"
