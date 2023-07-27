FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

# init workdir
RUN mkdir -p /build
WORKDIR /build

# install common tool & conda
RUN apt update && \
    apt install wget -y && \
    apt install git -y && \
    apt install ninja -y && \
    apt install vim -y && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    mkdir -p /opt/conda/envs/alpa && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
# echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
# echo "conda activate base" >> ~/.bashrc

ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LD_LIBRARY_PATH}
# install conda alpa env
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create --name alpa python=3.8 -y && \
    conda activate alpa && \
    apt install coinor-cbc -y && \
    pip3 install --upgrade pip && \
    pip3 install cupy-cuda113 && \
    pip3 install alpa && \
    pip3 install jaxlib==0.3.22+cuda113.cudnn820 -f https://alpa-projects.github.io/wheels.html && \
    pip3 install together_worker && \
    git clone https://github.com/togethercomputer/transformers_port && \
    cd transformers_port && pip install . && cd .. && rm -rf transformers_port && \
    pip3 install fastapi uvicorn omegaconf jinja2 einops && \
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113  && \
    pip3 install flash-attn && \
    cd /build && \
    git clone https://github.com/alpa-projects/alpa.git && \
    cd alpa/examples && \
    pip3 install -e . && \
    pip3 install sentencepiece && \
    pip3 install accelerate && \
    pip3 install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary && \
    echo "conda activate alpa" >> ~/.bashrc

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN wget https://together-distro-packages.s3.us-west-2.amazonaws.com/linux/x86_64/bin/together-node-latest -O /usr/local/bin/together-node && \
    chmod +x /usr/local/bin/together-node
COPY cfg-neoxt-local.yaml /home/user/cfg-neoxt.yaml
ENV PATH /opt/conda/condabin/conda/bin:$PATH
ENV HOME=/home/user
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app
COPY . /app
WORKDIR /app
