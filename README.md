# Quick_Deployment_HELM

To bring up a standalone node:

```console
docker run --rm --gpus device=2 \
 -v $PWD/.together:/home/user/.together \
 -it togethercomputer/native_hf_models /usr/local/bin/together-node start \
 --config /home/user/cfg-neoxt.yaml --color \
 --worker.service OpenChatTest --worker.model gpt-neoxt-v0.13
```

To bring up a standalone safety model:

```console
docker run --rm --gpus device=2 \
 -v $PWD/.together:/home/user/.together \
 -it togethercomputer/native_hf_models /usr/local/bin/together-node start \
 --config /home/user/cfg-neoxt.yaml --color \
 --worker.service SafetyTest --worker.model gpt-jt-safety
```
