#!/bin/bash
#SBATCH -N 1                     # number of nodes
#SBATCH -n 4                     # total number of processes
#SBATCH --ntasks-per-node 4      # tasks per node
#SBATCH -t 12:00:00             # wall time
#SBATCH -J gnmt                 # job name
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH -p dedicateq
#SBATCH --gres=gpu:4

set -x

SYSTEM=${SYSTEM:-"1xC4140"}
BENCHMARK_NAME="GNMT"
DATADIR=${DATADIR:-"/mnt/isilon/DeepLearning/database/mlperf/wmt16-en-de"}
LOGDIR=${LOGDIR:-"$PWD/results/$BENCHMARK"}
SHAREDIR=${SHAREDIR:-"/mnt/isilon/scratch"}
PREPROC_DATADIR=${PREPROC_DATADIR:-"/mnt/isilon/DeepLearning/database/mlperf/v0.6/gnmt"}
PREPROCESS=${PREPROCESS:-0}

DATE=`date '+%Y-%m-%d-%H-%M-%S'`
mkdir -p $LOGDIR
log_file=$LOGDIR/${SYSTEM}-${DATE}.log

num_nodes=$SLURM_JOB_NUM_NODES
gpus_per_node=$( echo $SLURM_TASKS_PER_NODE | cut -f 1 -d \( )

if [[ $PREPROCESS -eq 1 ]]; then
    singularity exec -w -B /cm/local/apps/cuda/libs/current:/mnt/driver -B $DATADIR:/data -B $PREPROC_DATADIR:/preproc_data \
		sandbox-pytorch-ngc19.05 bash /mnt/current/preproc_cmd.sh 
fi

(
hosts=( `scontrol show hostname |tr "\n" " "` )
pids=()
for node_id in `seq 0 $(($num_nodes-1))`; do
    if [[ $num_nodes -gt 1 ]]; then
        master_addr="$(hostname).ib.cluster"
        export MULTI_NODE=" --nnodes=$num_nodes --node_rank=$node_id --master_addr=$master_addr --master_port=4242 "
    fi
    srun -N 1 -n 1 -w ${hosts[$node_id]} \
    singularity exec -B /cm/local/apps/cuda/libs/current:/mnt/driver \
		     -B $DATADIR:/data -B $SHAREDIR:/mnt/shared -B $PREPROC_DATADIR:/preproc_data -B $PWD:/mnt/current \
        	     sandbox-pytorch-ngc19.05 bash /mnt/current/run_and_time.sh $SYSTEM $gpus_per_node $node_id &

    pids+=($!);
done

sleep 10
wait "${pids[@]}"
) |& tee $log_file

