Bootstrap: docker
From: neurodebian:latest

%help

    Container with packages on top of base 3.6 with anaconda

%files
  ./requirements.txt /requirements.txt

%post
  
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get -yq install \
    build-essential \
    wget \
    tmux \
    tree \
    vim \
    git \
    zip \
    unzip \
    graphviz \
    libsndfile1 \
    libasound2 \
    curl \
    ca-sertificates \
    apt-utils \
    postgresql-server.dev

  rm -rf /var/lib/apt/lists/*
  apt-get clean

  wget -c https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
    /bin/bash Anaconda3-2019.03-Linux-x86_64.sh -bfp /usr/local

  #Conda configuration of channels from .condarc file

  conda config --add channels defaults
  conda config --add channels conda-forge
  conda config --add channels pytorch
  conda config --add channels menpo
  conda update conda
  pip install --upgrade pip
  rm -rf /usr/local/lib/python3/site-packages/llvmlite*
  
  #Install environment
  rm -rf ~/anaconda3/lib/python3.6/site-packages/llvmlite*
  pip install -I -r requirements.txt
  pip uninstall matplotlib -y
  conda install nodejs pyqt
  conda install matplotlib --force-reinstall
  jupyter labextension install @jupyter-widgets/jupyterlab-manager
#  conda install xeus-python -c conda-forge
#  jupyter labextension install @jupyterlab/debugger
