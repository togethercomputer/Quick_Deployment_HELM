cd /home/fm/dev/Quick_Deployment_HELM

./together start &

/opt/conda/envs/alpa/bin/python serving_local_nlp_model.py --together_model_name together/gpt-neoxT-soda-v0.1 --hf_model_name Together/gpt-neoxT-20b --model_path /home/fm/helm_models/GPT-NeoXT-20B-soda-v0.1