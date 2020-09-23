import argparse
import pandas as pd
import sys
import os
import pathlib
import math
from tqdm import tqdm


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", default="data.csv", help="input csv file for processing")
    parser.add_argument("-c","--chunkscount", default=300, help="count of chunks to split")
    parser.add_argument("-o","--output", default='./', help="output directory for directories with batches")
    args = parser.parse_args()
    return args

def get_file_len(filepath):
    return len(pd.read_csv(filepath,usecols=[0,]))    

def get_chunksize(file_len,chunks_count):
    return int(math.ceil((file_len-1)/chunks_count))


def write_one_chunk_file(new_file_path,lines_to_write):
    print('writing '+str(new_file_path))
    with open(str(new_file_path.absolute()),'w') as g:
        for l in lines_to_write:
            g.write(l)
            g.write('\n')
            

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    
    input_file = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    if not input_file.exists():
        print("input file does not exists")
        exit(1)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True,exist_ok=True)
    
    file_len = get_file_len(input_file.absolute())
    chunksize = get_chunksize(file_len,int(args.chunkscount))
    
    print('chunksize is: '+str(chunksize))
    
    output_batch_dir = output_dir.absolute().joinpath(pathlib.Path('csv_chunks'))
    output_batch_dir.mkdir(parents=True,exist_ok=True) 
        
    infile_iterator = pd.read_csv(input_file.absolute(),
                                      converters={'bow': lambda x:' '.join([v for v in eval(x)]), 
                                                  'bow_with_context': lambda x:' '.join([v for v in eval(x)])},
                                       chunksize=chunksize)
        
    for chunk_num,data in tqdm(enumerate(infile_iterator)):
        data.index.name='docindex'
        filename = 'chunk_'+str(chunk_num)+'.csv'
        new_file_path = output_batch_dir.joinpath(pathlib.Path(filename))
        
        data[data.columns.difference(['docindex','text','formatted_msg_in_context'])].to_csv(new_file_path.absolute())
    
    