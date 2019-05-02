#!/usr/bin/pyhton
import sys
import argparse
import umap
import urllib
import numpy as np
import json
from json import loads, dumps



def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='paths to all the data vector files',type=str,nargs='+')
    parser.add_argument('-l', '--langs', help='langs of the vector files',type=str,nargs='+')
    parser.add_argument('-o', '--out', help='output file',type=str)
    return parser


def get_shared_keys(matrices):
    shared_keys = list(matrices[0].keys())
    for i in range(1,len(matrices)):
        new_keys = [k for k in list(matrices[i].keys()) if k in shared_keys]
        shared_keys = new_keys
    return shared_keys

def compute_umap(args):
    matrices_dict = {}
    for d,l in zip(args.data, args.langs):
        matrices_dict[l] = json.load(open(d,'r'))
    shared_keys = get_shared_keys(list(matrices_dict.values()))
    print('shared keys', len(shared_keys)) 

    data_umap = []
    for i in  shared_keys:
        for l in args.langs:
            matrix =  matrices_dict[l][str(i)]['encoding']
            f = np.asarray(matrix)
            f = f.flatten()
            data_umap.append(f)
    
    data_umap = np.array(data_umap)
    
    embedding = umap.UMAP(n_neighbors=10, min_dist=0.005, metric='correlation').fit_transform(data_umap)
    data = {'content':[]}
    failed = 0    
    for i in range(0,len(embedding),len(args.langs)):
        #try:
        data['content'].append({l:embedding[i+j].tolist() for j,l in enumerate(args.langs)})
        #except:
            #print('FAILED')
            #failed = failed + 1
    
    print('Failed', failed)
    with open(args.out,'w') as file:
        json.dump(data, file)
            


def main():
    parser = get_argparser()
    args = parser.parse_args()
    if len(args.data) != len(args.langs):
        print('Wrong parameters please check')
        sys.exit(1)
    compute_umap(args)

if __name__ == "__main__":
    main()







