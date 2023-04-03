FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install python=3.8
RUN conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch
RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install torchpack
RUN pip install pandas
RUN pip install -U scikit-learn
RUN pip install scikit-image
RUN pip install -U matplotlib
RUN pip install seaborn
RUN pip install einops


WORKDIR /root
COPY src/atrous_attn_vit /root/atrous_attn_vit
WORKDIR /root/atrous_attn_vit

ENV PATH=/usr/local/cuda-11.3/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-11.3
ENV FORCE_CUDA=1

RUN /bin/bash ./test_script.sh

