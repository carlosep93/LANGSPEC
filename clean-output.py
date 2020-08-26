import sys

hyp_dict = {}
i = 0
for line in sys.stdin:
    if line[0] == 'H':
        key = line.split('\t')[0].split('-')[-1] 
        hyp = line.split('\t')[-1].replace('\n','')
        if key != 'H':
            hyp_dict[key] = hyp
        else:
            hyp_dict[str(i)] = hyp
            i += 1
        

for i in range(len(hyp_dict)):
    print(hyp_dict[str(i)])
