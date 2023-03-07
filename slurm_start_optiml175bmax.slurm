#!/bin/bash
#SBATCH --job-name=together-alpa-OPT-175B
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --exclude=sphinx[1-3]
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --output=/afs/cs.stanford.edu/u/biyuan/fm/together/together-accelerate-OPT-iml-175B-max-%j.out
#SBATCH --error=/afs/cs.stanford.edu/u/biyuan/fm/together/together-accelerate-OPT-iml-175B-max-%j.err
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --gpus=a100:8

cd /nlp/scr2/nlp/fmStore/fm/dev/Quick_Deployment_HELM

nvidia-smi

# For docker mode:
docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host -v /nlp/scr2/nlp/fmStore/fm:/home/fm binhang/alpa /home/fm/dev/Quick_Deployment_HELM/start_local_optiml175bmax.sh