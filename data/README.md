# What is this?

This repository contain base image of singularity with snakemake, jupyterlab+ipywidgets, pytorch and some other libs, installed upon neurodebian linux repo.

# Contents of repo
```
.
├── notebooks          - notebookds with some demos
├── requirements.txt   - python requirements.txt with some libraries
├── Singularity        - singularity recipy for demo
└── README.md          - this document
```

# Where to start

- To install singularity on your local machine with ubuntu, do in terminal:

   `sudo apt-get install singularity-container` 

- To pull singularity image (if it's not already pulled somewhere): 
   
   `singularity pull shub://eugtsa/base_singularity:latest` 

- to pull singularity image: ``
- to run singularity image: `unset XDG_RUNTIME_DIR;singularity exec eugtsa-base_singularity-master-latest.simg jupyter lab`
- check out ipywidgets demo in `notebooks/1_ipywidgets_demo.ipynb`

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

- If you have a problem with running this singularity image on your local machine and traceback of error ends with:

  `File "/usr/local/lib/python3.7/os.py", line 211, in makedirs`
  
    `makedirs(head, exist_ok=exist_ok)`
    
  `File "/usr/local/lib/python3.7/os.py", line 221, in makedirs`
  
    `mkdir(name, mode)`
    
`PermissionError: [Errno 13] Permission denied: '/run/user'`

    
    This is known bug https://github.com/jupyter/notebook/issues/3826. Before running image, in same terminal do:
    
    `unset XDG_RUNTIME_DIR`
