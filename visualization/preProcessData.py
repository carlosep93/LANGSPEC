#!/usr/bin/pyhton
import sys
import umap
import urllib
import numpy as np
import json
from json import loads, dumps


if len(sys.argv) < 5:
	print ('Usage preProcessData languajeFile1 file1.json file1.txt languajeFile2 file2.json file2.txt #points')
	sys.exit()

lanF1 = sys.argv[1]
f1_json = open(sys.argv[2], 'r')
f1_txt = open(sys.argv[3], 'r')
lanF2 = sys.argv[4]
f2_json = open(sys.argv[5], 'r')
f2_txt = open(sys.argv[6], 'r')
n = int(sys.argv[7])
f1Matrices = json.load(f1_json)
f2Matrices = json.load(f2_json)
f1Phrases = f1_txt.read().split('\n')
f2Phrases = f2_txt.read().split('\n')

data = {}

data['type'] = 'SMAP'
data['content'] = []

umapData = []
umapPhrases = []


failed = 0
shared = [ k for k in f1Matrices.keys() if k in f2Matrices.keys()]
print(len(shared))
for i in  shared[:n]:
	i = int(i)
	print ('Encoding '+str(i))

	matrix =  f1Matrices[str(i)]['encoding']
	f1 = np.asarray(matrix)
	f1 = f1.flatten()

	umapData.append(f1)
	umapPhrases.append(f1Phrases[i])

	matrix = f2Matrices[str(i)]['encoding']
	f2 = np.asarray(matrix).flatten()

	umapData.append(f2)
	umapPhrases.append(f2Phrases[i])


ud = np.empty((n,umapData[0].shape[0]),dtype='float')
for i in range(len(umapData)-1):
	ud[i] = umapData[i]


embedding = umap.UMAP(n_neighbors=10,
					  min_dist=0.005,
					  metric='correlation').fit_transform(ud)



for i in range(0,len(embedding),2):
	try:
		data['content'].append({
		 	'f1':f1Phrases[i],
		 	'weights_f1':embedding[i].tolist(),
		 	'f2':f2Phrases[i],
		 	'weights_f2':embedding[i+1].tolist()
		})
	except:
		print('FAILED')
		failed = failed + 1


with open('data_umapp.json','w') as file:
	json.dump(data, file)


