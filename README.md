# Quick_Deployment_HELM

To deploy a new docker image, merge a PR to the main branch.

### To bring up a local REST server:

```console
mkdir -p .together/models
chmod 777 .together .together/models
docker run --rm --gpus device=0 \
  -v $PWD/.together:/home/user/.together \
  -e HF_HOME=/home/user/.together/models \
  -e HTTP_HOST=0.0.0.0 \
  -e SERVICE_DOMAIN=http \
  -p 5001:5001 \
  -it togethercomputer/native_hf_models /usr/bin/python3 serving_local_nlp_model.py --hf_model_name facebook/opt-350m
```

```console
curl -X POST -H 'Content-Type: application/json' http://localhost:5001/ -d '{"prompt": "Space robots"}'
```

```console
{"data": {"result_type": "language-model-inference", "choices": [{"text": " are a great way to get a lot of work done.", "index": 0, "finish_reason": "length"}], "raw_compute_time": 0.20327712898142636}}
```

### To bring up a standalone node:

```console
docker run --pull=always --rm --gpus device=2 \
 -v $PWD/.together:/home/user/.together \
 -it togethercomputer/native_hf_models /usr/local/bin/together-node start \
 --config /home/user/cfg-neoxt.yaml --color \
 --worker.service OpenChatTest --worker.model gpt-neoxt-v0.15
```

### To bring up a standalone node with retrieval:

```console
docker run --pull=always --rm --gpus device=2 \
 --add-host=host.docker.internal:host-gateway \
 -v $PWD/.together:/home/user/.together \
 -it togethercomputer/native_hf_models /usr/local/bin/together-node start \
 --config /home/user/cfg-neoxt-retrieval.yaml --color \
 --worker.service ock-faiss --worker.model gpt-neoxt-v0.15
```

### To bring up a standalone safety model:

```console
docker run --pull=always --rm --gpus device=2 \
 -v $PWD/.together:/home/user/.together \
 -it togethercomputer/native_hf_models /usr/local/bin/together-node start \
 --config /home/user/cfg-neoxt.yaml --color \
 --worker.service SafetyTest --worker.model gpt-jt-safety
```

### Start opt-350m in CPU on Mac laptop:

```console
~/together-node/build/together-node start --config ./cfg-opt-350m-docker-macos.yaml
```

### Start opt-350m in CPU on Linux:

```console
curl -O https://together-distro-packages.s3.us-west-2.amazonaws.com/linux/x86_64/bin/together-node-latest
chmod a+x ./together-node-latest
./together-node-latest start --config ./cfg-opt-350m-docker.yaml
```
