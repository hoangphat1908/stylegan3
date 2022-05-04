#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=32GB
#SBATCH --job-name=train
#SBATCH --output=train_sng_%j.out

singularity exec --nv \
--bind /scratch \
--overlay /scratch/pvn2005/my_env/stylegan.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
cd ..
python train.py --outdir=training-runs --data=datasets/SNGFaces-256x256.zip --cfg=stylegan2 --gpus=2 --batch=64 \
--gamma=0.2048 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --kimg=5000 \
--resume=pretrained/stylegan2-ffhq-256x256.pkl

"
