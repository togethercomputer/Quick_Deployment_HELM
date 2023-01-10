cd /home/fm/dev/Quick_Deployment_HELM

./together start &

/opt/conda/envs/alpa/bin/python serving_local_nlp_model.py --together_model_name together/codegen-16B-mono --hf_model_name Salesforce/codegen-16B-mono --model_path /home/fm/helm_models/hf_models/codegen-16B-mono