cd /home/fm/dev/Quick_Deployment_HELM

/opt/conda/envs/alpa/bin/python -m pip install accelerate

./together start &

/opt/conda/envs/alpa/bin/python serving_dist_accelerate_nlp_model.py --together_model_name together/galactica-120b --hf_model_name facebook/galactica-120b --model_path /home/fm/helm_models/hf_models/galactica-120b