#!/bin/bash
#SBATCH --job-name=together-hf-codegen-13b-mono
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --output=/afs/cs.stanford.edu/u/biyuan/fm/together/together-hf-codegen_13b_mono-%j.out
#SBATCH --error=/afs/cs.stanford.edu/u/biyuan/fm/together/together-hf-codegen_13b_mono-%j.err
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --gpus=a100:1


cd /nlp/scr2/nlp/fmStore/fm/dev/Quick_Deployment_HELM

echo "My host name:" $HOSTNAME

nvidia-smi

# For singularity mode: (assume alpa_general.sif in the current directory)
# singularity run --nv --bind /nlp/scr2/nlp/fmStore/fm:/home/fm alpa_general.sif /home/fm/dev/Quick_Deployment_HELM/start_alpa_bloom.sh

docker run --rm --gpus '"device=0"' --ipc=host -v /nlp/scr2/nlp/fmStore/fm:/home/fm binhang/alpa /home/fm/dev/Quick_Deployment_HELM/start_local_codegen_13b_mono.sh