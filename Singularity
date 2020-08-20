Bootstrap: docker
From: neurodebian:latest

%help

    Container with Anaconda 3 from neurodebian.
    This installation is based on Python 3.6

%post

  #Installing all dependencies
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get -yq install \
    build-essential \
    wget \
    git

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
