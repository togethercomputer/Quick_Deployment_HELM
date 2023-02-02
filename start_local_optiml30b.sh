cd /home/fm/dev/Quick_Deployment_HELM

./together start &

/opt/conda/envs/alpa/bin/python serving_local_nlp_model.py --together_model_name together/opt-iml-30b --hf_model_name facebook/opt-iml-30b --model_path /home/fm/helm_models/hf_models/opt-iml-30b
