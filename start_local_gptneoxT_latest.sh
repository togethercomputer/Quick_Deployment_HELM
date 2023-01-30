cd /home/fm/dev/Quick_Deployment_HELM

./together start &

#/opt/conda/envs/alpa/bin/python serving_local_nlp_model.py --together_model_name together/gpt-neoxT-20B-chat-latest --hf_model_name Together/gpt-neoxT-20b --model_path /home/fm/helm_models/hf_models/GPT-NeoXT-20B-soda-hc3-v0.3

/opt/conda/envs/alpa/bin/python serving_local_nlp_model.py --together_model_name together/gpt-neoxT-20B-chat-latest --hf_model_name Together/gpt-neoxT-20b --model_path /home/fm/helm_models/hf_models/GPT-NeoXT-20B-chat-v0.6