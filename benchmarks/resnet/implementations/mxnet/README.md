# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements
* [Singularity container](https://github.com/sylabs/singularity)
* [MXNet 19.05-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-mxnet)
* [OpenMPI compiler](https://www.open-mpi.org)

# 2. Directions
## Steps to download and verify data
Download the data using the following command:

Please download the dataset manually following the instructions from the [ImageNet website](http://image-net.org/download). We use non-resized Imagenet dataset, packed into MXNet recordio database. It is not resized and not normalized. No preprocessing was performed on the raw ImageNet jpegs.

For further instructions, see https://github.com/NVIDIA/DeepLearningExamples/blob/master/MxNet/Classification/RN50v1.5/README.md#prepare-dataset .

## Steps to build Singularity container

```
bash ./build_container.sh
```

## Steps to build OpenMPI compiler

The parallel ResNet-50 is implemented with Horovod library which uses MPI parallel programming model. So an MPI compiler is needed to run the benchmark. The MXNet 19.05 NGC container includes OpenMPI 3.1.3. The host needs to install an OpenMPI compiler with the version equal or greater than 3.1.3. By default OpenMPI 3.1.5 is used in the run script. You are free to change to other versions. 

```
wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.5.tar.gz
tar xzfv openmpi-3.1.5.tar.gz && cd openmpi-3.1.5
mkdir build && cd build
../configure --prefix=</path/to/install> --with-cuda=</path/to/cuda/toolkit>
make && make install
```

Then either create a module in your system, or add its bin and lib into PATH and LD_LIBRARY_PATH, respectively. 

## Steps to launch training

### Run with Slurm scheduler

Run on 1x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=1xC4140 sbatch -N 1 -n 4 run.slurm
```
Run on 2x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=2xC4140 sbatch -N 2 -n 8 run.slurm
```
Run on 4x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=4xC4140 sbatch -N 4 -n 16 run.slurm
```
Run on 8x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=8xC4140 sbatch -N 8 -n 32 run.slurm
```
Run on 16x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=16xC4140 sbatch -N 16 -n 64 run.slurm
```

### Run without Slurm scheduler

When no Slurm scheduler is used, make sure every participating node can access MPI compiler and library, and the host-\* file includes node names.

Run on 1x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=1xC4140
 
mpirun --allow-run-as-root --bind-to none --report-bindings -np 4 -npernode 4 -machinefile hosts-1 singularity exec -B /cm/local/apps/cuda/libs/current:/mnt/driver -B $DATADIR:/data -B $PWD:/mnt/current mxnet-ngc19.05-py3.sif bash /mnt/current/run_and_time.sh SYSTEM |& tee $LOGDIR/resnet_1xC4140.log
```

Run on 2x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=2xC4140 

mpirun --allow-run-as-root --bind-to none --report-bindings -np 8 -npernode 4 -machinefile hosts-2 singularity exec -B /cm/local/apps/cuda/libs/current:/mnt/driver -B $DATADIR:/data -B $PWD:/mnt/current mxnet-ngc19.05-py3.sif bash /mnt/current/run_and_time.sh $SYSTEM |& tee $LOGDIR/resnet_2xC4140.log
```

Run on 4x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=4xC4140 
mpirun --allow-run-as-root --bind-to none --report-bindings -np 16 -npernode 4 -machinefile hosts-4 singularity exec -B /cm/local/apps/cuda/libs/current:/mnt/driver -B $DATADIR:/data -B $PWD:/mnt/current mxnet-ngc19.05-py3.sif bash /mnt/current/run_and_time.sh $SYSTEM |& tee $LOGDIR/resnet_4xC4140.log
```

Run on 8x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=8xC4140

mpirun --allow-run-as-root --bind-to none --report-bindings -np 32 -npernode 4 -machinefile hosts-8 singularity exec -B /cm/local/apps/cuda/libs/current:/mnt/driver -B $DATADIR:/data -B $PWD:/mnt/current mxnet-ngc19.05-py3.sif bash /mnt/current/run_and_time.sh $SYSTEM |& tee $LOGDIR/resnet_8xC4140.log
```
Run on 16x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=16xC4140 
 
mpirun --allow-run-as-root --bind-to none --report-bindings -np 64 -npernode 4 -machinefile hosts-16 singularity exec -B /cm/local/apps/cuda/libs/current:/mnt/driver -B $DATADIR:/data -B $PWD:/mnt/current mxnet-ngc19.05-py3.sif bash /mnt/current/run_and_time.sh $SYSTEM |& tee $LOGDIR/resnet_16xC4140.log
```
