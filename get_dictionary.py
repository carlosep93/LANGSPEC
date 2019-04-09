
'''
From a list of files in the same language get a vocabulary file
'''

import argparse
from fairseq.data import dictionary
from fairseq.tokenizer import Tokenizer, tokenize_line


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora', help='paths to all the corpora files',type=str,nargs='+')
    parser.add_argument('-o', '--output', help='path to the output dictionary file', type=str)
    parser.add_argument('-n', '--nwords', help='number of words of the dictionary', type=int)
    return parser

def build_dictionary(filenames,output,nwords,workers=1,threshold=0,pfactor=8):
    d = dictionary.Dictionary()
    for filename in filenames:
        Tokenizer.add_file_to_dictionary(filename,d, tokenize_line,workers)
    d.finalize(
        threshold=threshold,
        nwords=nwords,
        padding_factor=pfactor,
    )
    d.save(output)

def log_data(args):
    print('*************************')
    print('* Corpora:',args.corpora)
    print('* Output:', args.output)
    print('* Number of tokens in dictionary', args.nwords)
    print('*************************')

def main():
    parser = get_argparser()
    args = parser.parse_args()
    log_data(args)
    build_dictionary(args.corpora,args.output,args.nwords)
    print('Dictionary created at', args.output)

if __name__ == "__main__":
    main()
