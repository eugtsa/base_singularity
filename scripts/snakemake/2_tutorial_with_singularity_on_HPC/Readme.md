Snakemake example usage with singularity on HPC cluster
=========================================

This is a basic snakemake example. 

1. To run one snakemake job, please do: `snakemake --use-singularity --cluster "sbatch -t {cluster.time} -p {cluster.partition} -N {cluster.nodes} --mem {cluster.memory}" --cluster-config cluster_config.yml --jobs 1 ds1_plot.jpg`
2. It would produce image, you could check it with `fim ds1_plot.jpg`
3. To tun complicated job, type in: `snakemake --use-singularity --cluster "sbatch -t {cluster.time} -p {cluster.partition} -N {cluster.nodes} --mem {cluster.memory}" --cluster-config cluster_config.yml --jobs 1 ds1_filtered_plot.jpg`
4. Try to run complicated job again - it should not run again, because all files are already created.
5. Modify file `ds1.csv` and try to run complicated job again. It should actually run because the input are newer than the output.


- `-p` argument to print actual commands snakemake running
- `-n` argument to build a plan, withou executing 
