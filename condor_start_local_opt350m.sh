nvidia-smi

echo "current directory:"
pwd

echo "files in current directory:"
ls

echo "files in /opt/conda/envs/:"
ls /opt/conda/envs/


# ./together start &

/opt/conda/envs/alpa/bin/python serving_local_nlp_model.py --together_model_name together/opt-350b --hf_model_name facebook/opt-350m


#nvidia-docker run --rm --gpus '"device=0"' --ipc=host -v ./:/home/fm binhang/alpa /opt/conda/envs/alpa/bin/python /home/fm/serving_local_nlp_model.py --together_model_name together/opt-350b --hf_model_name facebook/opt-350m