cd /home/fm/dev/Quick_Deployment_HELM

/opt/conda/envs/alpa/bin/python -m pip install accelerate

./together start &

/opt/conda/envs/alpa/bin/python serving_dist_accelerate_nlp_model.py --together_model_name together/opt-iml-175b-max --hf_model_name facebook/opt-iml-175b-max --model_path /home/fm/helm_models/hf_models/opt-iml-max