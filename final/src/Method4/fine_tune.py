# -*- coding: utf-8 -*-
import sys
import csv
import operator
from nltk.stem.porter import PorterStemmer

output_old = sys.argv[1]
output_new = output_old.split('.')[0] + '2.csv'

if output_old == 'test_est.csv':
	pos_name = 'pos_dict.csv'
else:
	pos_name = output_old.split('_')[0] + '_pos_dict.csv'

### Read pos_tagging result
pos_fh = open(pos_name)
reader = csv.DictReader(pos_fh)
pos_dict = {}
for idx,row in enumerate(reader):
	pos = row['POS'].split()
	pos_dict[row['word']] = {pos_count.split(':')[0]:int(pos_count.split(':')[1]) for pos_count in pos}

pos_fh.close()

### Calculate noun_ratio of each word
noun_ratio = {}
for word in pos_dict:
	noun_count = 0
	other_count = 0
	for pos in pos_dict[word]:
		if pos in ['NN', 'NNP', 'NNPS', 'NNS']:
			noun_count = noun_count + pos_dict[word][pos]
		else:
			other_count = other_count + pos_dict[word][pos]
	noun_ratio[word] = float(noun_count) / (noun_count + other_count)

## 2017/01/17: Read the probability of being a valid tag for all words
prob_fh = open('physics_prob.csv')
reader = csv.DictReader(prob_fh)
tag_prob = {}
for idx,row in enumerate(reader):
	tag_prob[row['word']] = float(row['prob'])

prob_fh.close()

## =========
## Fine tune
## =========
### Read the previous output results
fh_old = open(output_old)
reader = csv.DictReader(fh_old)
old = {}
for idx,row in enumerate(reader):
	tags = row['tags'].split()
	old[int(row['id'])] = tags

fh_old.close()
### Sort by id
old_sorted_by_id = sorted(old.items(), key=operator.itemgetter(0), reverse = False)

## [check and sort all tags]
all_old_tags = {}
for k,v in old.items():
	for tag in v:
		all_old_tags[tag] = all_old_tags.get(tag, 0) + 1

len(all_old_tags)

all_tags_sorted = sorted(all_old_tags.items(), key=operator.itemgetter(1), reverse = True)

### ---------------------------------------
### (1) Merge words into 2-grams or 3-grams
### ---------------------------------------
### (1-1) Merge words into 2-gram or 3-gram if they already exist
### in other questions at least 10 times.
### e.g., "relativity special theory"
###       ==> "special-relativity theory"
min_2gram_freq = 10
merged = {}
for idx,tags in old_sorted_by_id:
	if len(tags) > 1:
		temp_tags = []
		for tag1 in tags:
			tags_left = [t for t in tags if not t == tag1]
			for tag2 in tags_left:
				new_tag_2gram = tag1 + '-' + tag2
				if new_tag_2gram in all_old_tags and \
				   all_old_tags[new_tag_2gram] > min_2gram_freq:
					temp_tags.append(new_tag_2gram)
					break
			temp_tags.append(tag1)
	else:
		temp_tags = tags
	merged[idx] = temp_tags

### Remove tags that are already combined as 2-grams
### (If we combine 'special' and 'relativity' as 'special-relativity',
###  we can remove 'special' and 'relativity')
for idx,tags in merged.items():
	tags_copy = [t for t in tags if '-' not in t]
	tags_ngram = [t for t in tags if '-' in t]
	for tag1 in tags_copy:
		for tag2 in tags_ngram:
			if tag1 in tag2:
				tags.remove(tag1)
				break
	merged[idx] = tags

### (1-2) Merge two 2-grams ('A-B' and 'B-C') into one 3-gram ('A-B-C')
### ('quantum-field' and 'field-theory' can be merged into
###  'quantum-field-theory')
for idx,tags in merged.items():
	tags_ngram = [t for t in tags if '-' in t]
	if len(tags_ngram) > 1:
		for gram1 in tags_ngram:
			found = False
			gram1_splited = gram1.split('-')
			last_part = gram1_splited[len(gram1_splited)-1]
			for gram2 in [t for t in tags_ngram if t != gram1]:
				if last_part == gram2.split('-')[0]:
					merged_3grams = '-'.join(gram1_splited + gram2.split('-')[1:])
					if merged_3grams in all_old_tags:
						tags.append(merged_3grams)
					else:
						pass
					found = True
					break
			if found:
				break
	merged[idx] = tags

### Remove 2-grams that are already combined as 3-grams
### (If we combine 'quantum-field' and 'field-theory' as 'quantum-field-theory',
###  we can remove 'quantum-field' and 'field-theory')
for idx,tags in merged.items():
	tags_2gram = [t for t in tags if len(t.split('-')) == 2]
	tags_3gram = [t for t in tags if len(t.split('-')) == 3]
	for tag1 in tags_2gram:
		for tag2 in tags_3gram:
			if tag1 in tag2:
				tags.remove(tag1)
				break
	merged[idx] = tags

### --------------------------------------
### (2) Remove words that are impossible
###     to be noun (NN, NNP, NNPS, or NNS)
### 2017/01/02:
###     Remove words with noun_ratio < 0.5
### --------------------------------------
for idx,tags in merged.items():
	tags_copy = [t for t in tags if '-' not in t]
	for tag in tags_copy:
		if tag in noun_ratio:
			if noun_ratio[tag] < 0.5:
				tags.remove(tag)
		else:
			print tag
	merged[idx] = tags

### ----------------------------------------------------
### (3) STEM: aggregate similar tags into one single tag
###     For 2-grams or 3-grams:
###        choose the one with the highest frequency
###     For words:
###        Rule 1: Choose the one with the highest noun_ratio
###        (Choose 'entanglement' instead of 'entangled')
###        Rule 2: For simple nouns, use plural
###        (Choose 'neutrons' instead of 'neutron')
### ----------------------------------------------------
#### (3-1) Porter
stemmer1 = PorterStemmer()
tag_stem1 = {}
for tag,count in all_old_tags.items():
	temp_stem = stemmer1.stem(tag)
	if temp_stem in tag_stem1:
		tag_stem1[temp_stem][tag] = count
	else:
		tag_stem1[temp_stem] = {tag:count}

### TODO: which stemmer is better?
# ### (3-2) Snowball
# # from nltk.stem.snowball import SnowballStemmer
# # stemmer2 = SnowballStemmer("english")
# # tag_stem2 = [stemmer2.stem(tag) for tag,count in old_tags_sorted[0:50]]

### Make new results by stemming results
stemming_results = {}
for idx,row in merged.items():
	new_row = []
	if len(row) < 1:
		stemming_results[int(idx)] = []
		continue
	for tag in row:
		all_possible = {k:v for k,v in tag_stem1[stemmer1.stem(tag)].items()}
		#### Adjust all frequencies (counts) by multiplying them with noun_ratios
		for k,v in all_possible.items():
			if '-' in k:
				all_possible[k] = v
			else:
				all_possible[k] = v * noun_ratio[k]
		all_possible = sorted(all_possible.items(), key=operator.itemgetter(1), reverse = True)
		if len(all_possible) == 1:
			new_row.append(all_possible[0][0])
		#### Use plural if its frequency (count) is not too low
		#### (at least one fifth compare to the single version)
		elif (all_possible[0][0] + 's' == all_possible[1][0]) or \
		     (all_possible[0][0][-1] == 'y' and all_possible[0][0][:-1] + 'ies' == all_possible[1][0]):
			if all_possible[1][1] > 0 and float(all_possible[0][1]) / all_possible[1][1] < 5:
				new_row.append(all_possible[1][0]) ## plural
			else:
				new_row.append(all_possible[0][0]) ## single
		else:
			new_row.append(all_possible[0][0])
	stemming_results[int(idx)] = list(set(new_row))

### -------------------
### (4) tag probability
### -------------------
filtered_results = {}
for idx,row in stemming_results.items():
	new_row = []
	for tag in row:
		if ('-' in tag and len(tag) > 5) or (tag in tag_prob and tag_prob[tag] >= 0.75):
			new_row.append(tag)
	filtered_results[int(idx)] = list(set(new_row))


### -----------------------
### Write output file again
### -----------------------
output = open(output_new,'w')
writer = csv.writer(output, quoting = csv.QUOTE_ALL)
writer.writerow(['id','tags'])

keylist = filtered_results.keys()
keylist.sort()
for key in keylist:
	writer.writerow([key,' '.join(filtered_results[key])])

output.close()
