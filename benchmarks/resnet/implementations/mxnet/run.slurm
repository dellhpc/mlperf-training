#!/bin/bash
#SBATCH -N 1                     # number of nodes
#SBATCH -n 4                     # total number of processes
#SBATCH --ntasks-per-node 4      # tasks per node
#SBATCH -t 12:00:00             # wall time
#SBATCH -J image_classification # job name
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH -p dedicateq
#SBATCH --gres=gpu:4

set -x

SYSTEM=${SYSTEM:-"1xC4140"}
BENCHMARK_NAME="RESNET"
DATADIR=${DATADIR:-"/mnt/isilon/DeepLearning/database/mlperf/ilsvrc12_passthrough"}
LOGDIR=${LOGDIR:-"$PWD/results/$BENCHMARK"}
DATE=`date '+%Y-%m-%d-%H-%M-%S'`
mkdir -p $LOGDIR
log_file=$LOGDIR/${SYSTEM}-${DATE}.log

set +x
module load shared extra/openmpi/3.1.5
set -x

total_gpus=$SLURM_NTASKS
echo "total gpus: $SLURM_NTASKS "
echo "data dir: $DATADIR"

MPIRUN="mpirun --allow-run-as-root --bind-to none --report-bindings -np $total_gpus "

${MPIRUN} singularity exec -B /cm/local/apps/cuda/libs/current:/mnt/driver -B $DATADIR:/data -B $PWD:/mnt/current \
	mxnet-ngc19.05-py3.sif bash /mnt/current/run_and_time.sh $SYSTEM |& tee $log_file

