# A hard code example

cd /home/fm/dev/Quick_Deployment_HELM

# conda activate alpa

/opt/conda/envs/alpa/bin/ray start --head

./together start &

/opt/conda/envs/alpa/bin/python serving_dist_alpa_nlp_model.py --together_model_name together-alpa-bloom --alpa_model_name bloom --model_path /home/fm/helm_models/alpa_models/