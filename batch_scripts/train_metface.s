#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=32GB
#SBATCH --job-name=train
#SBATCH --output=train_metface_%j.out

singularity exec --nv \
--bind /scratch \
--overlay /scratch/pvn2005/my_env/stylegan.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
cd ..
python train.py --outdir=training-runs --data=datasets/metfaces-1024x1024.zip --cfg=stylegan2 --gpus=2 --batch=32 --gamma=2 --kimg=5000
"
