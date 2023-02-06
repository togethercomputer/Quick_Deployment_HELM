nvidia-smi

echo "current directory:"
pwd

echo "files in current directory:"
ls

unzip alpa.zip

chmod 777 together

./together start &

./alpa/bin/python serving_local_nlp_model.py --together_model_name together/opt-350m --hf_model_name facebook/opt-350m