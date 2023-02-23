#!/bin/bash
/opt/conda/envs/alpa/bin/ray start --head

/opt/conda/envs/alpa/bin/python serving_dist_alpa_nlp_model.py --alpa_model_name bloom --model_path /home/user/.together/models/