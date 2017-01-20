import numpy as np
import pandas as pd
import re
from nltk import FreqDist
import csv
import operator

target = sys.argv[1]
corpus_pd = pd.read_csv(target)
ids = corpus_pd.id.values.tolist()
out_old = sys.argv[2]
corpus_est = pd.read_csv(out_old)
tags_est = corpus_est.tags.values.tolist()

## Load clustering results
km_labels_0 = np.load('km_labels_0.npy')
km_labels_1 = np.load('km_labels_1.npy')
km_labels_2 = np.load('km_labels_2.npy')
km_labels_3 = np.load('km_labels_3.npy')
km_labels = {0:km_labels_0, 1:km_labels_1, 2:km_labels_2, 3:km_labels_3}

## Number of cluaters = 500
nb_cluster = len(set(km_labels_0))

corpus_common_tags = {}
for idx in xrange(len(ids)):
	corpus_common_tags[idx] = []

def frequent(context):
	freq = FreqDist(context)
	return freq

## Aggregate all clustering results for each question
## Take 10~15 minutes on macbook pro 2011 (4G-ram)
for k,labels in km_labels.items():
	for cl in xrange(nb_cluster):
		temp_est = []
		# n_items = 0
		for idx in xrange(len(labels)):
			if (labels[idx] == cl):
				# n_items = n_items + 1
				if not type(tags_est[idx]) is float:
					for tag in re.split(' ', tags_est[idx]):
						temp_est.append(tag)
				corpus_common_tags[idx].append(temp_est)

## Take 5~10 minutes on macbook pro 2011 (4G-ram)
for idx,common_tags in corpus_common_tags.items():
	corpus_common_tags[idx] = frequent([tag for tag_list in common_tags for tag in tag_list])

## Read tag probability
prob_fh = open('physics_prob.csv')
reader = csv.DictReader(prob_fh)
tag_prob = {}
for idx,row in enumerate(reader):
	tag_prob[row['word']] = float(row['prob'])

prob_fh.close()

## Debug only
corpus_common_tags_backup = corpus_common_tags
corpus_common_tags = corpus_common_tags_backup

## Write new output
out_new = out_old.split('_')[0] + '_est3.csv'
output_fh = open(out_new,'w')
writer = csv.writer(output_fh, quoting = csv.QUOTE_ALL)
writer.writerow(['id','tags'])
for i in xrange(len(tags_est)):
	## Find the top-5 cluster tags
	for tag in corpus_common_tags[i]:
		if '-' in tag:
			corpus_common_tags[i][tag] *= 2
		else:
			if tag in tag_prob:
				corpus_common_tags[i][tag] *= tag_prob[tag]
			else:
				corpus_common_tags[i][tag] = 0
	if type(tags_est[i]) is float:
		# tags_new = corpus_common_tags[i].max()
		corpus_common_tags_sorted = dict(sorted(corpus_common_tags[i].iteritems(), key=operator.itemgetter(1), reverse=True)[:2])
		tags_new = ' '.join(corpus_common_tags_sorted.keys())
	# elif len(re.split(' ', tags_est[i])) == 1:
	else:
		### Copy
		tags_new = tags_est[i][:]
		### Append the "cluster-representation" word
		corpus_common_tags_sorted = dict(sorted(corpus_common_tags[i].iteritems(), key=operator.itemgetter(1), reverse=True)[:1])
		for tag,count in corpus_common_tags_sorted.items():
			if count > 100 and (not tag in tags_new):
				tags_new = tags_new + ' ' + tag
	# else:
	# 	### Copy
	# 	tags_new = tags_est[i][:]
	# 	### Append the "cluster-representation" word
	# 	corpus_common_tags_sorted = dict(sorted(corpus_common_tags[i].iteritems(), key=operator.itemgetter(1), reverse=True)[:1])
	# 	for tag,count in corpus_common_tags_sorted.items():
	# 		if count > 100 and (not tag in tags_new):
	# 			tags_new = tags_new + ' ' + tag
	writer.writerow([ids[i], tags_new])

output_fh.close()

