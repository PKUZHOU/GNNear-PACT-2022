FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update  && apt-get install -y gfortran build-essential \
    && apt-get install -y curl git ninja-build && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/miniconda3/bin:${PATH} CONDA_PREFIX=/opt/miniconda3

RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -p /opt/miniconda3 -b && \
    rm Miniconda3-py37_4.8.3-Linux-x86_64.sh && \
    conda install pytorch=1.10.0 cudatoolkit=10.2 cudnn -c pytorch -y && \
    conda install conda-verify conda-build mkl-include cmake ninja -c anaconda -y && \
    conda clean -afy


RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html 
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html 
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html 
RUN pip install --no-cache-dir  matplotlib==3.3.4 ogb==1.3.1 pynvml==11.0.0 tqdm==4.61.1

RUN pip install dgl-cu102 dglgo -f https://data.dgl.ai/wheels/repo.html 
RUN pip install torch-geometric==1.7.2

WORKDIR /workspace