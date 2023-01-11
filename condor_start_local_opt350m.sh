nvidia-smi

echo "print current directory:"
ls

./together start &

#singularity run --nv --bind ./:/home/fm alpa_general.sif /opt/conda/envs/alpa/bin/python /home/fm/serving_local_nlp_model.py --together_model_name together/opt-350b --hf_model_name facebook/opt-350m


nvidia-docker run --rm --gpus '"device=0"' --ipc=host -v ./:/home/fm binhang/alpa /opt/conda/envs/alpa/bin/python /home/fm/serving_local_nlp_model.py --together_model_name together/opt-350b --hf_model_name facebook/opt-350m