# 1. Problem

Object Detection 

## Requirements
* [Singularity container](https://github.com/sylabs/singularity)
* [PyTorch 19.05-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)

# 2. Directions
## Steps to download and verify data

```
cd reference/single_stage_detector/
source download_dataset.sh
```

## Steps to build Singularity container

```
bash ./build_container.sh
```

## Steps to launch training

### Run with Slurm scheduler

Run on 1x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=1xC4140 sbatch -N 1 -n 4 ./run.slurm
```
Run on 2x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=2xC4140 sbatch -N 2 -n 8 ./run.slurm
```
Run on 4x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=4xC4140 sbatch -N 4 -n 16 ./run.slurm
```
Run on 8x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=8xC4140 sbatch -N 8 -n 32 ./run.slurm
```
Run on 16x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=16xC4140 sbatch -N 16 -n 64 ./run.slurm
```
### Hyperparameter settings

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time.sh`.

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ImageNet (from torchvision).

# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.23

### Evaluation frequency
The model is evaluated at epochs 33, 44, 50, 55, 61 and 66.

### Evaluation thoroughness
All the images in COCO 2017 val data set.
