# What is this?

This repository contain base image of singularity with snakemake, jupyterlab+ipywidgets, pytorch and some other libs, installed upon neurodebian linux repo.

# Contents of repo
```
.
├── notebooks          - directory with notebookds with some demos
├── data               - sample data to check out jupyter lab
├── scripts            - directory with scripts for demo
├── requirements.txt   - python requirements.txt with some libraries
├── Singularity        - singularity recipy for demo
└── README.md          - this document
```

# How to start: base steps (needed for all later steps)

- install singularity: https://sylabs.io/guides/3.4/user-guide/installation.html#install-the-debian-ubuntu-package-using-apt

- clone this repo: `$ git clone https://github.com/eugtsa/base_singularity.git;cd base_singularity`

- run prepare.sh script in this repo root, it would download nescessary singularity image (about 4.5Gb) and unpack data: `$ ./prepare.sh`

# How to start with Jupyter lab

- do base steps (find above)

- start jupyter lab from singularity container, run in terminal: `$ singularity exec base_singularity.sif jupyter lab`

- copy link from terminal to your browser

- check out ipywidgets demo in `notebooks/ipywidgets/1_Widgets Basics.ipynb`

# How to start with snakemake

- do base steps (find above)

- start jupyter lab from singularity container, run in terminal: `$ singularity exec base_singularity.sif jupyter lab`

- copy link from terminal to your browser

- check out Readme for example in: `scripts/snakemake/1_tutorial/Readme.md`

# How to sun snakemake with singularity

- do base steps (find above)

- install snakemake: "$ pip install snakemake"

- check out Readme for example in: `scripts/snakemake/2_tutorial_with_singularityReadme.md`

# How to sun snakemake with singularity on cluster

- login on cluster (this step is cluster-specific)

- run tmux session: `$ tmux`

- clone this repo: `$ git clone https://github.com/eugtsa/base_singularity.git;cd base_singularity`

- run prepare.sh script in this repo root, it would download nescessary singularity image and unpack data: `$ ./prepare.sh`

- run jupyter lab: `$ jupyter lab --no-browser --port 18799`

- install snakemake: `$ pip install snakemake --user`

- check out Readme for example in: `scripts/snakemake/2_tutorial_with_singularity_on_HPC/Readme.md`

- alternatively, run bigger example: `scripts/snakemake/3_bigger_example/Readme.md`

# Singularity

We use singularity (https://sylabs.io/guides/3.2/user-guide/)

## Base image

[![https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg](https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg)](https://singularity-hub.org/collections/4679)

Recipe for image is located here:
`https://github.com/eugtsa/base_singularity`.
Singularity repo is located on github because it has native integrations with singularity hub. It allows us to 
automatically build singularity recipes and store resulting images on singularity hub.

After build this image is available through singularity hub: https://www.singularity-hub.org/collections/4679

Image is based on neurodebian (http://neuro.debian.net/) and include common scientific libraries.



## Known bugs

- If you have problem with running our image on your local machine and traceback of error starts with:

  `File "/usr/local/lib/python3.7/os.py", line 211, in makedirs`
  
    `makedirs(head, exist_ok=exist_ok)`
    
  `File "/usr/local/lib/python3.7/os.py", line 221, in makedirs`
  
    `mkdir(name, mode)`
    
`PermissionError: [Errno 13] Permission denied: '/run/user'`

    
    This is known bug https://github.com/jupyter/notebook/issues/3826. Before running image, in same terminal do:
    
    `unset XDG_RUNTIME_DIR`
