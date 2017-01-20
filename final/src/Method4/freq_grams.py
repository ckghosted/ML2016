# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk import FreqDist, pos_tag, ngrams
import csv
import re
import sys
import operator
import string

# ======================================
# Part I: Read file and define functions
# ======================================
## corpus_file: 'test.csv' (for physics), 'biology.csv', 'cooking.csv', etc.
corpus_file = sys.argv[1]
## Read the corpus
corpus = pd.read_csv(corpus_file, encoding = 'utf-8')

run_pos_tag = False
if run_pos_tag:
	if corpus_file == 'test.csv':
		pos_dict_file = 'pos_dict.csv'
	else:
		pos_dict_file = corpus_file.split('.')[0] + '_pos_dict.csv'


## output_name: 'test.csv' ==> 'test_est.csv', etc.
output_name = corpus_file.split('.')[0] + '_est.csv'

meaning_less = ['p','would','could','via','emp','two','must','make',
                'e','c','using','r','vs','versa','based','three']
meaning_less = meaning_less + ['non', 'need', 'number']
my_stopwords = set(stopwords.words('english')).union(meaning_less)

punct = set(string.punctuation)

def clear_stopwords(context):
	letters = re.sub("[^a-zA-Z]", " ", context).lower().split()
	clear = [c for c in letters if c not in my_stopwords]
	return clear

def remove_html(context):
	cleaner = re.compile('<.*?>')
	clean_text = re.sub(cleaner,'',context)
	return clean_text

## Find all already-existed 2-grams
def find_all_2grams(context):
	grams = re.findall("[a-zA-Z]+\\-[a-zA-Z]+", context, flags=0)
	return grams

## Find all already-existed 3-grams
def find_all_3grams(context):
	grams = re.findall("[a-zA-Z]+\\-[a-zA-Z]+\\-[a-zA-Z]+", context, flags=0)
	return grams

## Make n-grams (we use n = 2 or 3) by joining consecutive words
def make_n_gram(context, n):
	letters = re.sub("[^a-zA-Z]", " ", context).lower().split()
	if n < 3:
		clear = [c for c in letters if c not in my_stopwords]
	else:
		clear = letters
	context = ' '.join(word for word in clear)
	## Remove punctuation and make it all lowercase
	context = ''.join(ch for ch in context if ch not in punct)
	n_grams = ngrams(context.split(), n)
	return ['-'.join(g) for g in n_grams]

## Make frequency distributions from a list
def frequent(context):
	freq = FreqDist(context)
	return freq

# ===========================================
# Part II: Read the whole corpus the 1st time
# ===========================================
df = open(corpus_file)
reader_for_all = csv.DictReader(df)
all_1grams_list = []
all_2grams_list = []
all_3grams_list = []
pos_dict = {}
for idx,row in enumerate(reader_for_all):
	## (1) all_1grams_dict
	title = clear_stopwords(row['title'])
	all_1grams_list.append([t for t in title])
	content = remove_html(row['content'])
	content = clear_stopwords(content)
	all_1grams_list.append([t for t in content])
	## (2) all_2grams_dict
	keep_hyphen = find_all_2grams(row['title'])
	if len(keep_hyphen):
		all_2grams_list.append([t for t in keep_hyphen])
	content = remove_html(row['content'])
	keep_hyphen = find_all_2grams(content)
	if len(keep_hyphen):
		all_2grams_list.append([t for t in keep_hyphen])
	## (3) all_3grams_dict
	keep_hyphen = find_all_3grams(row['title'])
	if len(keep_hyphen):
		all_3grams_list.append([t for t in keep_hyphen])
	keep_hyphen = find_all_3grams(content)
	if len(keep_hyphen):
		all_3grams_list.append([t for t in keep_hyphen])
	### 2016/12/31: Do POS tagging once and save the results as 'pos_dict.csv'
	### after this for loop
	# (4) POS tagging
	if run_pos_tag:
		title_pos = nltk.pos_tag(re.sub("[^a-zA-Z]", " ", remove_html(re.sub("\\n", " ", row['title']))).lower().split())
		content_pos = nltk.pos_tag(re.sub("[^a-zA-Z]", " ", remove_html(re.sub("\\n", " ", row['content']))).lower().split())
		pos_list = title_pos + content_pos
		for word, pos in pos_list:
			if word in pos_dict:
				pos_dict[word][pos] = pos_dict[word].get(pos, 0) + 1
			else:
				pos_dict[word] = {pos: 1}

df.close()

all_1grams_dict_tmp = frequent([w.lower() for w_list in all_1grams_list for w in w_list])
### Remove bad 1-grams that are too short (e.g., less than 3 characters)
all_1grams_dict = {}
for (k,v) in all_1grams_dict_tmp.items():
	if len(k) > 3:
		all_1grams_dict[k] = v

all_2grams_dict_tmp = frequent([w.lower() for w_list in all_2grams_list for w in w_list])
## Remove silly 2-grams:
all_2grams_list_removed = []
### The same 1st and 2nd token (e.g., 'time-time' or 'charge-charge')
for gram in all_2grams_dict_tmp:
	splited = gram.split('-')
	if splited[0] == splited[1]:
		all_2grams_list_removed.append(gram)
### Remove bad 2-grams that are:
### (1) From all_2grams_list_removed above;
### (2) Too short (e.g., less than 5 characters)
all_2grams_dict = {}
for (k,v) in all_2grams_dict_tmp.items():
	if not k in all_2grams_list_removed and len(k) > 5:
		all_2grams_dict[k] = v

all_3grams_dict_tmp = frequent([w.lower() for w_list in all_3grams_list for w in w_list])
## Remove silly 3-grams:
all_3grams_list_removed = []
### Two or more stopwords (e.g., 'action-at-a' or 'up-to-date')
for gram in all_3grams_dict_tmp:
	count_stopword = 0
	for word in gram.split('-'):
		if word in my_stopwords:
			count_stopword = count_stopword + 1
	if count_stopword > 1:
		all_3grams_list_removed.append(gram)
### The same 1st and 3rd token (e.g., 'one-to-one' or 'step-by-step')
for gram in all_3grams_dict_tmp:
	splited = gram.split('-')
	if splited[0] == splited[2]:
		all_3grams_list_removed.append(gram)
### Remove bad 3-grams that are:
### (1) From all_3grams_list_removed above;
### (2) Too rare (e.g., only appear 1 time in the whole corpus);
### (3) Too short (e.g., less than 7 characters)
all_3grams_dict = {}
for (k,v) in all_3grams_dict_tmp.items():
	if not k in all_3grams_list_removed and v > 1 and len(k) > 7:
		all_3grams_dict[k] = v

### Save pos_tagging result
if run_pos_tag:
	pos_dict_output = open(pos_dict_file,'w')
	writer = csv.writer(pos_dict_output, quoting = csv.QUOTE_ALL)
	writer.writerow(['word','POS'])
	for key in pos_dict:
		writer.writerow([key,' '.join(pos + ':' + str(count) for pos,count in pos_dict[key].items())])
	pos_dict_output.close()

# ============================================
# Part III: Read the whole corpus the 2nd time
# ============================================
df = open(corpus_file)
reader = csv.DictReader(df)
output = open(output_name,'w')
writer = csv.writer(output, quoting = csv.QUOTE_ALL)
writer.writerow(['id','tags'])
for idx,row in enumerate(reader):
	title = clear_stopwords(row['title']) ## return list
	content = remove_html(row['content'])
	content = clear_stopwords(content)
	common = set(content).intersection(title)
	freq_title = frequent(title)
	freq_content = frequent(content)
	all_2grams = make_n_gram(row['title'], 2) + \
	            make_n_gram(remove_html(row['content']), 2)
	all_3grams = make_n_gram(row['title'], 3) + \
	             make_n_gram(remove_html(row['content']), 3)
	freq_2grams = frequent(all_2grams)
	freq_3grams = frequent(all_3grams)
	### Add the 1st 2-gram into "common" if its frequency > 1,
	### add the 2nd 2-gram into "common" if its frequency > 2,
	### add the 3nd 2-gram into "common" if its frequency > 3.
	### (Add at-most three 2-grams)
	tags_2gram = []
	min_gram_freq = 1
	for (k,v) in freq_2grams.most_common(10):
		if 'http' in k:
			continue
		if min_gram_freq > 3:
			break
		if v > min_gram_freq and k in all_2grams_dict:
			tags_2gram.append(k)
			tags_2gram = list(set(tags_2gram))
		min_gram_freq = min_gram_freq + 1
	## Add the 1st 3_gram if its frequency > 1
	tags_3gram = []
	min_gram_freq = 1
	for (k,v) in freq_3grams.most_common(10):
		if 'http' in k or min_gram_freq > 1:
			break
		if v > min_gram_freq and k in all_3grams_dict:
			tags_3gram.append(k)
			tags_3gram = list(set(tags_3gram))
		min_gram_freq = min_gram_freq + 1
	## Consider duplication in 2-grams and 3-grams
	## (e.g., if we have 'faster-than-light', we don't need 'faster-light')
	if len(tags_2gram):
		tags_2gram_copy = [x for x in tags_2gram]
		for gram in tags_2gram_copy:
			for gram2 in tags_3gram:
				if gram in gram2:
					tags_2gram.remove(gram)
	tags_ngram = tags_2gram + tags_3gram
	## Consider duplication in words and n-grams
	## (e.g., if we have 'string-theory', we can remove both 'string' and 'theory' if any)
	if len(tags_ngram):
		common_copy = [x for x in common]
		for word in common_copy:
			for gram in tags_ngram:
				if word in gram:
					common.discard(word)
	common = common.union(tags_ngram)
	## Remove tags with less than 4 characters
	common_copy = [x for x in common]
	for tag in common_copy:
		if len(tag) < 4:
			common.remove(tag)
	## Write output file
	if len(common) == 0:
		## If no common frequent word nor candidate n-grams,
		## choose words from title directly (sometimes help)
		temp = []
		for t in title:
			if t not in meaning_less and len(t) > 3:
				temp.append(t)
		writer.writerow([row['id'],' '.join(temp)])
	else:
		writer.writerow([row['id'],' '.join(common)])

df.close()
output.close()

