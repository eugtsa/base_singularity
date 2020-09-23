import argparse
import pandas as pd
import sys
import pathlib
from tqdm import tqdm

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", default="input.csv", help="input csv file")
    parser.add_argument("-o","--output", default='./', help="output file for vw format")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    input_file = pathlib.Path(args.input)
    output_file = pathlib.Path(args.output)
    
    if not input_file.exists():
        print("input file does not exists")
        exit(1)
    
    # check and create parent dir
    if not output_file.parents[0].exists():
        output_file.parents[0].mkdir(parents=True,exist_ok=True)
    

    data = pd.read_csv(input_file.absolute(),
                usecols = ["docindex","bow_with_context",],
                index_col=0)
    
    with open(output_file.absolute(),'w') as f:
        for tup in data.itertuples():
            vw_line = 'doc_'+str(tup[0])+' |bow_with_context '+str(tup[1])+'\n'
            f.write(vw_line)