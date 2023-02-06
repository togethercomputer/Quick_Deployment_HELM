echo "This should be inside container"

nvidia-smi

echo "current directory:"
pwd

echo "files in current directory:"
ls

echo "files in /opt/conda/envs directory:"
ls /opt/conda/envs

echo "ping Google:"
ping google.com -c 5

# chmod 777 together

# ./together start &

# /opt/conda/envs/alpa/bin/python serving_local_nlp_model.py --together_model_name together/opt-350m --hf_model_name facebook/opt-350m
