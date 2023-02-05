#!/bin/bash
/opt/conda/envs/alpa/bin/ray start --head

/opt/conda/envs/alpa/bin/python serving_dist_alpa_nlp_model.py --together_model_name together/bloom --alpa_model_name bloom --model_path /home/user/.together/models/