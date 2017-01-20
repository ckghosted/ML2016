import pandas as pd
import csv
import copy
from collections import Counter
del_tagset = set()
del_dict = {}
origin_data = pd.read_csv('../merged_resluts/result.csv')
data = origin_data.tags.values.tolist()
data = [x.split() if not isinstance(x, float) else [] for x in data]
for i, tags in enumerate(data):
	for j, tag in enumerate(tags):
		tagsplit = tag.split('-')
		if len(tagsplit) == 2:
			word = tagsplit[0] + tagsplit[1]
			reverse = tagsplit[1] + '-' + tagsplit[0]
			if tagsplit[0] == tagsplit[1]:
				del data[i][j]
			elif word in tags or word[:-1] in tags:
				if word in tags:
					w = word
				else:
					w = word[:-1]
				del_tagset.add(tag)
				if tag not in del_dict:
					del_dict[tag] = w
				del data[i][j]
			elif reverse + 's' in tags or reverse in tags:
				del data[i][j]
		elif len(tagsplit) == 3:
			def check(a, b):
				try:
					x = tags.index(a + '-' + b)
				except ValueError:
					x = None
				if x is not None:
					del data[i][x]
			check(tagsplit[0], tagsplit[1])
			check(tagsplit[1], tagsplit[2])
			check(tagsplit[0], tagsplit[2])
print del_tagset, del_dict
all_tagset = Counter([tag for tags in data for tag in tags])
bigram_min_times = 3
replace_dash = {(v if all_tagset[k] >= all_tagset[v] * bigram_min_times else k) \
	: (k if all_tagset[k] >= all_tagset[v] * bigram_min_times else v) for k, v in del_dict.iteritems()}
tag_mincount = 25
bad_tagset = set([k for k, v in all_tagset.iteritems() if v <= tag_mincount])
for i, tags in enumerate(data):
	for j, tag in enumerate(tags):
		if tag in set(bad_tagset):
			data[i][j] = '.'
		elif tag in replace_dash:
			data[i][j] = replace_dash[tag]
	data[i] = [w for w in data[i] if w != '.']

data = [' '.join(x) for x in data]
origin_data['tags'] = data
print origin_data
origin_data.to_csv('../merged_resluts/tuned.csv', index=False, quoting=csv.QUOTE_ALL)
