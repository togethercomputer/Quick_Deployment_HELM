cd /home/fm/dev/Quick_Deployment_HELM

./together start &

/opt/conda/envs/alpa/bin/python serving_local_nlp_model.py --together_model_name together/chip_20B_instruct_alpha --hf_model_name chip_20B_instruct_alpha --model_path /home/fm/helm_models/hf_models/chip_20B_instruct_alpha
