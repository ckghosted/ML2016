import pandas as pd
import csv
import os
x = 0
path = '../cluster_results/result' + str(x) + '.csv'
while os.path.isfile(path):
	if x == 0:
		results = pd.read_csv(path)
	else:
		results = pd.concat([results, pd.read_csv(path)])
	x += 1
	path = '../cluster_results/result' + str(x) + '.csv'
results = results.sort_values(['id'], ascending=True)
print results
results.to_csv('../merged_resluts/result.csv', index=False, quoting=csv.QUOTE_ALL)
