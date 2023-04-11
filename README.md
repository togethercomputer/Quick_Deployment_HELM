# Quick_Deployment_HELM

To deploy a new docker image, merge a PR to the main branch.

To bring up a standalone node:

```console
docker run --pull=always --rm --gpus device=2 \
 -v $PWD/.together:/home/user/.together \
 -it togethercomputer/native_hf_models /usr/local/bin/together-node start \
 --config /home/user/cfg-neoxt.yaml --color \
 --worker.service OpenChatTest --worker.model gpt-neoxt-v0.15
```

To bring up a standalone node with retrieval: (Which will HTTP GET [http://host.docker.internal:5001](retrieval_plugin.py) by default. To change add `--plugin_url` to [cfg.yaml](cfg-neoxt-retrieval-local.yaml).

```console
docker run --pull=always --rm --gpus device=2 \
 --add-host=host.docker.internal:host-gateway \
 -v $PWD/.together:/home/user/.together \
 -it togethercomputer/native_hf_models /usr/local/bin/together-node start \
 --config /home/user/cfg-neoxt-retrieval.yaml --color \
 --worker.service ock-faiss --worker.model gpt-neoxt-v0.15
```

To bring up a standalone safety model:

```console
docker run --pull=always --rm --gpus device=2 \
 -v $PWD/.together:/home/user/.together \
 -it togethercomputer/native_hf_models /usr/local/bin/together-node start \
 --config /home/user/cfg-neoxt.yaml --color \
 --worker.service SafetyTest --worker.model gpt-jt-safety
```

Start opt-350m in CPU on Mac laptop:

```console
~/together-node/build/together-node start --config ./cfg-opt-350m-docker-macos.yaml \
  --worker.image_pull none
```

Start opt-350m in CPU on Linux:

```console
curl -O https://together-distro-packages.s3.us-west-2.amazonaws.com/linux/x86_64/bin/together-node-latest
chmod a+x ./together-node-latest
./together-node-latest start --config ./cfg-opt-350m-docker.yaml
```
