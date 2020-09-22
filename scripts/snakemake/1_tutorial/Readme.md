Snakemake example
=========================================

This is a basic snakemake example. 


1. To run one snakemake job, please do: `snakemake ds1_plot.jpg --cores all`
2. It would produce image, you could check it out `ds1_plot.jpg`
3. To produce dag: `snakemake --dag ds1_plot.jpg --cores all | dot -Tpng > dag.png`
3. To tun complicated job, type in: `snakemake ds1_filtered_plot.jpg --cores all`
4. Try to run complicated job again - it should not run again, because all files are already created.
5. Modify file `ds1.csv` and try to run complicated job again. It should actually run because the input are newer than the output.


Use `-p` argument to print actual commands snakemake running
