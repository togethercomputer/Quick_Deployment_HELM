#!/bin/bash
#SBATCH --job-name=together-alpa-OPT-2.7B
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/afs/cs.stanford.edu/u/biyuan/fm/together/together-alpa-OPT-2.7B-%j.out
#SBATCH --error=/afs/cs.stanford.edu/u/biyuan/fm/together/together-alpa-OPT-2.7B-%j.err
#SBATCH --partition=jag-standard
#SBATCH --account=nlp
#SBATCH --gpus=3090:2


cd /nlp/scr2/nlp/fmStore/fm/dev/Quick_Deployment_HELM

# For singularity mode: (assume alpa_general.sif in the current directory)
# singularity run --nv --bind /nlp/scr2/nlp/fmStore/fm:/home/fm alpa_general.sif /home/fm/dev/Quick_Deployment_HELM/start_alpa_opt2.7b.sh

# For docker mode:
docker run --rm --gpus '"device=0,1"' --ipc=host -v /nlp/scr2/nlp/fmStore/fm:/home/fm binhang/alpa /home/fm/dev/Quick_Deployment_HELM/start_alpa_opt2.7b.sh