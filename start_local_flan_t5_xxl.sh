cd /home/fm/dev/Quick_Deployment_HELM


./together start &


/opt/conda/envs/alpa/bin/python serving_local_nlp_model.py --together_model_name together/flan-t5-xxl --hf_model_name google/flan-t5-xxl --model_path /home/fm/helm_models/hf_models/flan-t5-xxl