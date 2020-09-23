Snakemake example usage with singularity on HPC cluster
=========================================

This is a basic snakemake example. 

1. To run one snakemake job, please do: `snakemake --use-singularity --cluster "sbatch -t {cluster.time} -p {cluster.partition} -N {cluster.nodes} --mem {cluster.memory}" --cluster-config cluster_config.yml --jobs 1 proj.vw`
2. It would produce result file, you could check it out: `proj.vw`


- `-p` argument to print actual commands snakemake running
- `-n` argument to build a plan, without executing 
