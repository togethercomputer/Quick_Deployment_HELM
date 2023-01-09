#!/bin/bash
#SBATCH --job-name=together-alpa-OPT-175B
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --exclude=sphinx[1-3]
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --output=/afs/cs.stanford.edu/u/biyuan/fm/together/together-alpa-OPT-175B-%j.out
#SBATCH --error=/afs/cs.stanford.edu/u/biyuan/fm/together/together-alpa-OPT-175B-%j.err
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --gpus=a100:8


cd /nlp/scr2/nlp/fmStore/fm/dev/Quick_Deployment_HELM

nvidia-smi

# For singularity mode: (assume alpa_general.sif in the current directory)
# singularity run --nv --bind /nlp/scr2/nlp/fmStore/fm:/home/fm alpa_general.sif /home/fm/dev/Quick_Deployment_HELM/start_alpa_opt175b.sh

# For docker mode:
docker run --rm --gpus '"device=0,1,2,3,4,5,6,7"' --ipc=host -v /nlp/scr2/nlp/fmStore/fm:/home/fm binhang/alpa /home/fm/dev/Quick_Deployment_HELM/start_alpa_opt175b.sh