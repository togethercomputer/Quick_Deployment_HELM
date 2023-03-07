# Quick_Deployment_HELM

To bring up a standalone node:

```console
ls .together/models/gpt-neoxt-v0.13/config.json
docker run --rm --gpus device=2 \
 -v $PWD/.together:/root/.together \
 -it togethercomputer/native_hf_models /usr/local/bin/together-node start \
 --config /root/cfg-neoxt.yaml --color \
 --worker.service OpenChatTest --worker.model gpt-neoxt-v0.13
```
