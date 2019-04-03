import numpy as np
from sklearn import linear_model
import json
import sys

if len(sys.argv) < 3:
	print ('Usage preProcessData infile outfile')
	sys.exit()

with open(sys.argv[1],'r') as f:
	data = json.load(f)


def normalize(a):
	a = np.array(a)
	row_sums = a.sum(axis=1)
	new_matrix = a / row_sums[:, np.newaxis]
	return new_matrix

def remove_bias(a,b):
	difs = a - b
	m = difs.mean(axis=0)
	print(m)
	return b - m

x = []
y = []
for p in data['content']:
	x.append(p['weights_f1'])
	y.append(p['weights_f2'])

x = np.array(x)
y = np.array(y)

ransac = linear_model.RANSACRegressor()
ransac.fit(x, y)

ransac_y = ransac.predict(y)
#ransac_y = remove_bias(x,ransac_y)

data_ransac = {'type':'SMAP', 'content':[]}

for i in range(len(data['content'])):
	c = data['content'][i]
	c['weights_f1'] = x[i].tolist()
	c['weights_f2'] = ransac_y[i].tolist()
	data_ransac['content'].append(c)

with open(sys.argv[2],'w') as f:
	json.dump(data_ransac,f)



