import json

embs = 'embeddings-europarl-big3-de.json'
out = 'embs.de.vec'

with open(embs) as e, open(out,'w') as o:
    data = json.load(e)
    print(len(list(data.keys())),len(list(data.values())[0]),file=o)
    for k,v in data.items():
        v = str(v).replace(',',' ').replace('\n','')
        print(k.replace(' ','_'),v[1:-1],file=o)


