# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk import FreqDist
import nltk
from collections import defaultdict, Counter
import csv
import os
import re
import sys
import operator
import random

#########
from util import get_tag_and_probs, prob_bool, get_algo_set
#########

algo_sets = set(get_algo_set())
file_index = 0 
file_name = '../clusters/num' + str(file_index) + '.csv'

while os.path.isfile(file_name):
	NN_tagset, NN_bi_tagset, NN_tri_tagset = get_tag_and_probs(file_index)
	output_name = '../cluster_results/result' + str(file_index) + '.csv'
	# file_name = './test_string_theory.csv'
	# file_name = './test.csv'
	# output_name = 'output_4.csv'
	# output_name = 'output_test_best.csv'

	# from subprocess import check_output
	# print(check_output(["ls", "../input"]).decode("utf8"))
	# df = open('./test.csv')

	# my_stopwords = set([u'using', u'use', u'file', u'files', u'get', u'way', u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves', u'its', u'before', u'with', u'had', u'should', u'to', u'only', u'under', u'ours', u'has', u'do', u'them', u'his', u'very', u'they', u'not', u'during', u'now', u'him', u'nor', u'did', u'these', u't', u'each', u'where', u'because', u'doing', u'theirs', u'some', u'are', u'our', u'ourselves', u'out', u'what', u'for', u'below', u'does', u'above', u'between', u'she', u'be', u'we', u'after', u'here', u'hers', u'by', u'on', u'about', u'of', u'against', u's', u'or', u'own', u'into', u'yourself', u'down', u'your', u'from', u'her', u'whom', u'there', u'been', u'few', u'too', u'themselves', u'was', u'until', u'more', u'himself', u'that', u'but', u'off', u'herself', u'than', u'those', u'he', u'me', u'myself', u'this', u'up', u'will', u'while', u'can', u'were', u'my', u'and', u'then', u'is', u'in', u'am', u'it', u'an', u'as', u'itself', u'at', u'have', u'further', u'their', u'if', u'again', u'no', u'when', u'same', u'any', u'how', u'other', u'which', u'you', u'who', u'most', u'such', u'why', u'a', u'don', u'i', u'having', u'so', u'the', u'yours', u'once'])
	meaning_less = ['p','would','could','via','emp','two','must','make',
						 'e','c','using','r','vs','versa','based','three']
	meaning_less = meaning_less + ['non', 'need', 'number']
	my_stopwords = set(stopwords.words('english')).union(meaning_less)
	my_stopwords |= algo_sets
	import string
	punct = set(string.punctuation)

	def clear_stopwords(context):
		letters = re.sub("[^a-zA-Z]", " ", context).lower().split()
		# stopword = set(stopwords.words('english'))
		clear = [c for c in letters if c not in my_stopwords]
		return clear

	def find_all_dashed(context):
		dashed = re.findall("[a-zA-Z]+\\-[a-zA-Z]+", context, flags=0)
		return dashed

	def find_all_dashed2(context):
		dashed = re.findall("[a-zA-Z]+\\-[a-zA-Z]+\\-[a-zA-Z]+", context, flags=0)
		return dashed

	# clear_stopwords('What is your simplest explanation of the string theory?')
	# clear_stopwords_keep_dash('What is your simplest explanation of the string-theory?')

	from nltk import ngrams
	def make_n_gram(context, n):
		letters = re.sub("[^a-zA-Z]", " ", context).lower().split()
		if n < 3:
			clear = [c for c in letters if c not in my_stopwords]
		else:
			clear = letters
		context = ' '.join(word for word in clear)
		# print context
		## Remove punctuation and make it all lowercase
		context = ''.join(ch for ch in context if ch not in punct)
		n_grams = ngrams(context.split(), n)
		return ['-'.join(g) for g in n_grams]

	# make_n_gram('What is your simplest explanation of the string theory?', 2)
	# make_n_gram('Controllable faster-than-light phase velocity', 3)

	# title2 = ''.join(ch for ch in sentence2.lower() if ch not in exclude)
	# 	n_grams = ngrams(title2.split(), n)
	# 	for grams in n_grams:
	# 		title.append("-".join(grams))

	def remove_html(context):
		## remove the content between <code> and </code> first
		# cleaner = re.compile('<code>.*?</code>')
		# context = re.sub(cleaner,'',context)
		cleaner = re.compile('<.*?>')
		clean_text = re.sub(cleaner,'',context)
		return clean_text

	test_string = '''<p>This is a question that has been posted at many different forums, I thought maybe someone here would have a better or more conceptual answer than I have seen before:</p>

	<p>Why do physicists care about representations of Lie-groups? For myself, when I think about a representation that means there is some sort of group acting on a vector space, what is the vector space that this Lie group is acting on? </p>

	<p>Or is it that certain things have to be invariant under a group action?
	maybe this is a dumb question, but i thought it might be a good start...</p>

	<p>To clarify, I am specifically thinking of the symmetry groups that people think about in relation to the standard model. I do not care why it might be a certain group, but more how we see the group acting, what is it acting on? etc.</p>'''
	# remove_html(re.sub("\\n", " ", test_string))
	# remove_html('<p>How do you go about it if the light beam has different polarizations in different parts of the transverse plane? One example is a <a href=""http://en.wikipedia.org/wiki/Radial_polarization"">radially polarized</a> beam. More generally, is there a good technique for sampling the local polarization (which might be linear, elliptical, or circular, anywhere on the <a href=""http://en.wikipedia.org/wiki/Poincare_sphere"">Poincaré sphere</a>) in one transverse plane?</p>')

	# def remove_code(context):
	# 	cleaner = re.compile('<code>.*?</code>')
	# 	clean_text = re.sub(cleaner,'',context)
	# 	return clean_text

	# remove_code('<pre><code>aW1 := c[z, 0] == If[z &gt; 0, 0, Ca]    rW2 := d Derivative[1, 0][c][0, t] - v c[0, t] == v Ca</code></pre>')

	def frequent(context):
		freq = FreqDist(context)
		return freq

	## 2016/12/22: POS tagging
	# temp_context = 'Velocity of Object from electromagnetic field'
	# nltk.pos_tag(re.sub("[^a-zA-Z]", " ", temp_context).lower().split())
	# nltk.pos_tag(clear_stopwords(temp_context))
	# temp_context = 'What is your simplest explanation of the string theory?'
	# nltk.pos_tag(re.sub("[^a-zA-Z]", " ", temp_context).lower().split())
	# nltk.pos_tag(clear_stopwords(temp_context))
	# temp_context = 'What is your simplest explanation of the string theories?'
	# nltk.pos_tag(re.sub("[^a-zA-Z-]", " ", temp_context).lower().split())
	# nltk.pos_tag(clear_stopwords(temp_context))
	### ==> we should not clear stopwords before POS tagging
	## Use dictionary to record all possible POS for each word
	# temp_context = test_string
	# pos_list = nltk.pos_tag(re.sub("[^a-zA-Z-]", " ", temp_context).lower().split())
	# pos_dict = {}
	# for word, pos in pos_list:
	# 	if word in pos_dict:
	# 		pos_dict[word][pos] = pos_dict[word].get(pos, 0) + 1
	# 	else:
	# 		pos_dict[word] = {pos: 1}


	# from nltk import ngrams
	# sentence = 'this is a foo bar sentences and i want to ngramize it'
	# n = 2
	# n_grams = ngrams(sentence.split(), n)
	# for grams in n_grams:
	# 	print grams



	### ==================================
	### Read thw whole corpus the 1st time
	### ==================================
	df = open(file_name)
	reader_for_all = csv.DictReader(df)
	all_words_ = []
	all_dashed_ = []
	all_dashed2_ = []
	pos_dict = {}
	for idx,row in enumerate(reader_for_all):
		# print row['id']
		## (1) all_words
		title = clear_stopwords(row['title']) ## return list
		all_words_.append([t for t in title])
		content = remove_html(row['content'])
		content = clear_stopwords(content)
		all_words_.append([t for t in content])
		## (2) all_dashed
		title_keep_dash = find_all_dashed(row['title']) ## return list
		# print title_keep_dash
		if len(title_keep_dash):
			all_dashed_.append([t for t in title_keep_dash])
		content = remove_html(row['content'])
		content_keep_dash = find_all_dashed(content)
		# print content_keep_dash
		if len(content_keep_dash):
			all_dashed_.append([t for t in content_keep_dash])
		## (3) all_dashed2
		title_keep_dash2 = find_all_dashed2(row['title']) ## return list
		# print title_keep_dash
		if len(title_keep_dash2):
			all_dashed2_.append([t for t in title_keep_dash2])
		content_keep_dash2 = find_all_dashed2(content)
		# print content_keep_dash
		if len(content_keep_dash2):
			all_dashed2_.append([t for t in content_keep_dash2])
		## 2016/12/31: Do POS tagging once and save the results as 'pos_dict.csv'
		## (4) POS tagging
		# title_pos = nltk.pos_tag(re.sub("[^a-zA-Z]", " ", remove_html(re.sub("\\n", " ", row['title']))).lower().split())
		# content_pos = nltk.pos_tag(re.sub("[^a-zA-Z]", " ", remove_html(re.sub("\\n", " ", row['content']))).lower().split())
		# pos_list = title_pos + content_pos
		# for word, pos in pos_list:
		# 	if word in pos_dict:
		# 		pos_dict[word][pos] = pos_dict[word].get(pos, 0) + 1
		# 	else:
		# 		pos_dict[word] = {pos: 1}

	# all_words = set([w.lower() for w_list in all_words for w in w_list])
	all_words = frequent([w.lower() for w_list in all_words_ for w in w_list])
	# print all_words.most_common(100)
	# print all_words['spacetime']
	# print '===================\n'
	all_dashed = frequent([w.lower() for w_list in all_dashed_ for w in w_list])
	# print all_dashed.most_common(100)
	# print all_dashed['space-time']
	# for k in all_dashed:
	# 	if len(k) < 4:
	# 		print k
	all_dashed2_tmp = frequent([w.lower() for w_list in all_dashed2_ for w in w_list])
	# print all_dashed2.most_common(100)
	## Remove silly 3-grams:
	all_dashed2_removed = []
	### (1) two or more stopwords (e.g., 'action-at-a' or 'up-to-date')
	for gram in all_dashed2_tmp:
		count_stopword = 0
		for word in gram.split('-'):
			if word in my_stopwords:
				count_stopword = count_stopword + 1
		if count_stopword > 1:
			all_dashed2_removed.append(gram)
	### (2) the same 1st and 3rd token (e.g., 'one-to-one' or 'step-by-step')
	for gram in all_dashed2_tmp:
		splited = gram.split('-')
		if splited[0] == splited[2]:
			all_dashed2_removed.append(gram)

	all_dashed2 = {}
	for (k,v) in all_dashed2_tmp.items():
		if not k in all_dashed2_removed and v > 1 and len(k) > 7:
			all_dashed2[k] = v

	# all_dashed2_sorted = sorted(all_dashed2.items(), key=operator.itemgetter(1), reverse = True)
	# print all_dashed2_sorted[0:10]

	# if 'special-relativity' in pos_dict:
	# 	print pos_dict['special-relativity']
	# else:
	# 	print '"special-relativity" not in pos_dict'

	# for word in pos_dict:
	# 	print word
	# 	print pos_dict[word]

	### Save pos_tagging result
	# pos_dict_output = open('pos_dict.csv','w')
	# writer = csv.writer(pos_dict_output, quoting = csv.QUOTE_ALL)
	# writer.writerow(['word','POS'])
	# for key in pos_dict:
	# 	writer.writerow([key,' '.join(pos + ':' + str(count) for pos,count in pos_dict[key].items())])

	# pos_dict_output.close()

	### Read pos_tagging result
	pos_fh = open('pos_dict.csv')
	reader = csv.DictReader(pos_fh)
	pos_dict = {}
	for idx,row in enumerate(reader):
		pos = row['POS'].split()
		pos_dict[row['word']] = {pos_count.split(':')[0]:int(pos_count.split(':')[1]) for pos_count in pos}

	pos_fh.close()

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

	### ==================================
	### Read thw whole corpus the 2nd time
	### ==================================
	short_2gram = {}
	short_tags = {}
	df.close()

	df = open(file_name)
	reader = csv.DictReader(df)
	preds = defaultdict(list)
	output = open(output_name,'w')
	writer = csv.writer(output, quoting = csv.QUOTE_ALL)
	writer.writerow(['id','tags'])
	# count = 0
	for idx,row in enumerate(reader):
		# print '\n'
		# count = count + 1
		# if count > 2:
			# break
		# print idx
		# print row
		title = clear_stopwords(row['title']) ## return list
		# title = title + make_n_gram(row['title'], 2)
		# print title
		content = remove_html(row['content'])
		content = clear_stopwords(content)
		# content = content + make_n_gram(row['content'], 2)
		# if row['id'] == '37':
		# 	print row['content']
		# 	print '===================\n'
		# 	content = remove_html(row['content'])
		# 	content = clear_stopwords(content)
		# 	print content
		# 	print '===================\n'
		# 	letters = re.sub("[^a-zA-Z-]", " ", remove_html(re.sub("\\n", " ", row['content']))).lower().split()
		# 	print letters
		# 	print '===================\n'
		# 	clear = [c for c in letters if c not in my_stopwords]
		# 	print clear
		# 	print '===================\n'
		# 	pos_tagging = nltk.pos_tag(letters)
		# 	print pos_tagging
		# 	print '===================\n'
		# print content
		freq_title = frequent(title)
		# print freq_title.most_common(5)
		freq_content = frequent(content)
		# print freq_content.most_common(5)
		all_grams = make_n_gram(row['title'], 2) + \
						make_n_gram(remove_html(row['content']), 2)
		all_grams2 = make_n_gram(row['title'], 3) + \
						 make_n_gram(remove_html(row['content']), 3)
		# print 'ID = %s' % row['id']
		# print row['content']
		# print remove_html(row['content'])
		freq_grams = frequent(all_grams)
		# print freq_grams.most_common(3)
		freq_grams2 = frequent(all_grams2)
		# print freq_grams2.most_common(3)
		# if row['id'] == '185':
			# print freq_grams.most_common(5)
		preds[row['id']].append(' '.join(title[:]))
		# if row['id'] == '73':
			# print 'preds:'
			# print preds
		#writer.writerow([row['id'],' '.join(title[:3])])
		common = set(content).intersection(title)

	#########
		c = list(title + content)
		if idx == 12:
			print c
		NN_unitags = []
		NN_bitags_pri = []
		NN_bitags = []
		NN_tritags = []
		NN_bicheck = {}

		for idx, x in enumerate(c):
			if x in NN_tagset:
				NN_unitags += [x]
			elif x[:len(x)-1] in NN_tagset:
				NN_unitags += [x[:len(x)-1]]
			for k in NN_bi_tagset:
				if x == k[0] or x[:len(x)-1] == k[0]:
					if k not in NN_bicheck:
						NN_bicheck[k] = [1, 0, idx]
					else:
						NN_bicheck[k][0] += 1
						NN_bicheck[k][2] = idx
				elif x == k[1] or x[:len(x)-1] == k[1]:
					if k not in NN_bicheck:
						NN_bicheck[k] = [0, 1, -2]
					else:
						NN_bicheck[k][1] += 1
						if NN_bicheck[k][2] == idx - 1:
							NN_bicheck[k][2] = -1
			if idx >= 2:
				test = (c[idx-2], c[idx-1], x)
				if test in NN_tri_tagset:
					NN_tritags += [test[0] + '-' + test[1] + '-' + test[2]]
		NN_unitags = Counter(NN_unitags)
		NN_unitags = [x for x, c in NN_unitags.most_common(4) if c >= 2]
		for k, v in NN_bicheck.iteritems():
			if v[0] >= 1 and v[1] >= 1:
				word = k[0] + '-' + k[1]
				w_inverse = k[1] + '-' + k[0]
				if v[2] == -1:
					if k[0] in NN_unitags and k[1] in NN_unitags:
						NN_unitags.remove(k[0])
						NN_unitags.remove(k[1])
				if w_inverse not in NN_bitags_pri:
					if prob_bool(NN_bi_tagset[k]):
						if v[2] == -1:
							NN_bitags_pri += [word]
						else:
							NN_bitags += [word]
					else:
						if v[2] == -1:
							NN_bitags_pri += [(word + 's')]
						else:
							NN_bitags += [(word + 's')]
		if NN_tritags:
			NN_tritags = random.sample(set(NN_tritags), 1)
		else:
			NN_tritags = []
		NN_tritags = [x for x in NN_tritags if prob_bool(0.5)]
		NN_bitags = [x for x in NN_bitags if prob_bool(0.1)]
		NN_unitags = [x if prob_bool(NN_tagset[x]) else x + 's' for x in NN_unitags]
		NN_unitags = [x for x in NN_unitags if prob_bool(0.2)]
		all_tags = [x for x in (NN_tritags + NN_bitags_pri + NN_unitags + NN_bitags) if len(x) >= 5]
	#########
		# print common
		# common = common.union([k for (k,v) in freq_grams.most_common(1) if v > 1])
		freq_grams_v = freq_grams.most_common(1)[0][1]
		freq_grams_max = [k for (k,v) in freq_grams.items() if v == freq_grams_v]
		# print freq_grams_max
		## Add the 1st 2_gram if its frequency > 1, the 2nd 2_gram if its frequency > 2, ...
		gram_tags = []
		min_gram_freq = 1
		for (k,v) in freq_grams.most_common(10):
			if 'http' in k:
				continue
			if min_gram_freq > 3:
				break
			if v > min_gram_freq and k in all_dashed:
				gram_tags.append(k)
				gram_tags = list(set(gram_tags))
			min_gram_freq = min_gram_freq + 1
		if len(gram_tags):
			gram_tags_copy = [x for x in gram_tags]
			for dashed in gram_tags_copy:
				if len(dashed) < 5:
					# short_2gram[dashed] = short_2gram.get(dashed, 0) + 1
					gram_tags.remove(dashed)
				## Replace 'space-time' by 'spacetime' if 'spacetime' is more frequent
				elif all_words[re.sub('-','',dashed)] > all_dashed[dashed]:
					gram_tags.remove(dashed)
					gram_tags.append(re.sub('-','',dashed))
		# print gram_tags
		## Add the 1st 3_gram if its frequency > 1
		gram_tags2 = []
		min_gram_freq = 1
		for (k,v) in freq_grams2.most_common(10):
			if 'http' in k or min_gram_freq > 1:
				break
			if v > min_gram_freq and k in all_dashed2:
				gram_tags2.append(k)
				gram_tags2 = list(set(gram_tags2))
			min_gram_freq = min_gram_freq + 1
		# print gram_tags2
		if len(gram_tags2):
			gram_tags2_copy = [x for x in gram_tags2]
			for dashed in gram_tags2_copy:
				if len(dashed) < 7:
					# short_2gram[dashed] = short_2gram.get(dashed, 0) + 1
					gram_tags2.remove(dashed)
		# print gram_tags2
		## Consider duplication in 2-grams and 3-grams
		## (e.g., if we have 'faster-than-light', we don't need 'faster-light')
		if len(gram_tags):
			gram_tags_copy = [x for x in gram_tags]
			for gram in gram_tags_copy:
				for gram2 in gram_tags2:
					if gram in gram2:
						gram_tags.remove(gram)
		gram_tags = gram_tags + gram_tags2
		# print gram_tags
		# gram_tags = [k for (k,v) in freq_grams.most_common(1) if v > 1 and 'http' not in k and (k in all_dashed or k in all_dashed2)]
		if len(gram_tags):
			# print common
			# print gram_tags
			## If we have 'string-theory', we can remove both 'string' and 'theory' if any
			common_copy = [x for x in common]
			for word in common_copy:
				for gram in gram_tags:
					if word in gram:
						common.discard(word)
	##########
		for x in gram_tags:
			tt = x.split('-')
			if len(tt) == 2:
				if tt[1] + '-' + tt[0] in NN_bitags_pri:
					gram_tags.remove(x)
	##########
		common = common.union(gram_tags)
		# print row['id']
		# print common
		common_copy = [x for x in common]
		for tag in common_copy:
			# print '	', tag
			if len(tag) < 4:
				short_tags[tag] = short_tags.get(tag, 0) + 1
				common.remove(tag)
			### 2016/12/31: don't remove words that are not noun here
			### (we need 'special' and 'relativity' to form 'special-relativity'
			## POS tagging: remove words that are impossible to be noun (NN, NNP, NNPS, or NNS)
			# elif tag in pos_dict:
			# 	pos_list = [p for p in pos_dict[tag]]
			# 	if not ('NN' in pos_list or 'NNP' in pos_list or 'NNPS' in pos_list or 'NNS' in pos_list):
			# 		common.remove(tag)
			# 	else:
			# 		print '		', pos_dict[tag]
		# print common
		# print '============'
		# print '\n'
	########
		max_taglen_NN = 2
		if len(all_tags) > max_taglen_NN:
			all_tags = all_tags[:max_taglen_NN]
		all_tags += list(common)
		max_taglen = 6
		if len(all_tags) > max_taglen:
			all_tags = all_tags[:max_taglen]
		common = set(all_tags)
	########
		if len(common) == 0:
			# print 'len(common) == 0 for ID = %s' % row['id']
			#for t in title:
			#	if t not in meaning_less and len(t) > 3:
			#		temp.append(t)
			#print('ID : {} , Title : {}'.format(idx+1,title))
			temp = []
			writer.writerow([row['id'],' '.join(temp)])
			# writer.writerow([row['id'],' '.join([tag + ':' + pos for tag in temp for pos in pos_dict.get(tag, '?')])])
		else:
			writer.writerow([row['id'],' '.join(common)])
			# writer.writerow([row['id'],' '.join([tag + ':' + pos for tag in common for pos in pos_dict.get(tag, '?')])])
			#writer.writerow([row['id'],' '.join(set(content).intersection(title))])


	df.close()
	output.close()

	## =========
	## Fine tune
	## =========
	### Read the output produced by the above codes
	fh_old = open(output_name)
	reader = csv.DictReader(fh_old)
	old = {}
	for idx,row in enumerate(reader):
		tags = row['tags'].split()
		old[int(row['id'])] = tags

	fh_old.close()
	### Sort by id
	old_sorted_by_id = sorted(old.items(), key=operator.itemgetter(0), reverse = False)
	# old_sorted_by_id[0:50]

	# for idx,tags in old_sorted_by_id[0:50]:
	# 	print idx
	# 	for tag in tags:
	# 		if tag in pos_dict:
	# 			print tag, pos_dict[tag]
	# 		else:
	# 			print tag, '(not in pos_dict)'

	## [check and sort all tags]
	all_old_tags = {}
	for k,v in old.items():
		for tag in v:
			all_old_tags[tag] = all_old_tags.get(tag, 0) + 1

	len(all_old_tags)

	all_tags_sorted = sorted(all_old_tags.items(), key=operator.itemgetter(1), reverse = True)
	# all_tags_sorted[0:50]

	### -------------------------------------
	### (1) Merge words into 2-gram or 3-gram
	### e.g., "relativity special theory"
	###		 ==> "special-relativity theory"
	### -------------------------------------
	min_2gram_freq = 10
	merged = {}
	# for idx,tags in old.items():
	for idx,tags in old_sorted_by_id:
		# print idx, tags
		if len(tags) > 1:
			temp_tags = []
			for tag1 in tags:
				tags_left = [t for t in tags if not t == tag1]
				# print tag1, tags_left
				for tag2 in tags_left:
					new_tag_2gram = tag1 + '-' + tag2
					# print '   2-gram:', new_tag_2gram
					if new_tag_2gram in all_old_tags and all_old_tags[new_tag_2gram] > min_2gram_freq:
						# print '      ADD!'
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
		# print idx, tags
		tags_copy = [t for t in tags if '-' not in t]
		tags_dashed = [t for t in tags if '-' in t]
		# print '   ', tags_copy
		# print '   ', tags_dashed
		for tag1 in tags_copy:
			for tag2 in tags_dashed:
				if tag1 in tag2:
					# print tag1, 'in', tag2
					# print tag1, tag2
					tags.remove(tag1)
					break
		merged[idx] = tags

	### Merge two 2-grams ('A-B' and 'B-C') into one 3-gram ('A-B-C')
	### ('quantum-field' and 'field-theory' can be merged into
	###  'quantum-field-theory')
	for idx,tags in merged.items():
		tags_dashed = [t for t in tags if '-' in t]
		if len(tags_dashed) > 1:
			for dashed in tags_dashed:
				found = False
				dashed_splited = dashed.split('-')
				last_part = dashed_splited[len(dashed_splited)-1]
				for dashed2 in [t for t in tags_dashed if t != dashed]:
					if last_part == dashed2.split('-')[0]:
						merged_3grams = '-'.join(dashed_splited + dashed2.split('-')[1:])
						if merged_3grams in all_old_tags:
							tags.append(merged_3grams)
						else:
							pass
							# print idx, merged_3grams
						found = True
						break
				if found:
					break
		merged[idx] = tags

	### Remove 2-grams that are already combined as 3-grams
	### (If we combine 'quantum-field' and 'field-theory' as 'quantum-field-theory',
	###  we can remove 'quantum-field' and 'field-theory')
	for idx,tags in merged.items():
		# print idx, tags
		tags_2gram = [t for t in tags if len(t.split('-')) == 2]
		tags_3gram = [t for t in tags if len(t.split('-')) == 3]
		# print '   ', tags_copy
		# print '   ', tags_dashed
		for tag1 in tags_2gram:
			for tag2 in tags_3gram:
				if tag1 in tag2:
					# print tag1, 'in', tag2
					# print tag1, tag2
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
	# from nltk.stem import *
	#### (3-1) Porter
	from nltk.stem.porter import *
	stemmer1 = PorterStemmer()
	tag_stem1 = {}
	for tag,count in all_old_tags.items():
		temp_stem = stemmer1.stem(tag)
		if temp_stem in tag_stem1:
			tag_stem1[temp_stem][tag] = count
		else:
			tag_stem1[temp_stem] = {tag:count}

	# for stem in tag_stem1:
	# 	if len(tag_stem1[stem]) > 10:
	# 		print stem, tag_stem1[stem]
	# 		for tag in tag_stem1[stem]:
	# 			if tag in noun_ratio:
	# 				print '   ', tag, ':', noun_ratio[tag]
	# 			else:
	# 				print '   ', tag, '(not in noun_ratio)'

	# ### (3-2) Snowball
	# # from nltk.stem.snowball import SnowballStemmer
	# # stemmer2 = SnowballStemmer("english")
	# # tag_stem2 = [stemmer2.stem(tag) for tag,count in old_tags_sorted[0:50]]

	### Make new results by stemming results
	stemming_results = {}
	for idx,row in merged.items():
		# print idx
		new_row = []
		if len(row) < 1:
			stemming_results[int(idx)] = []
			continue
		for tag in row:
			all_possible = {k:v for k,v in tag_stem1[stemmer1.stem(tag)].items()}
			# print '   ', all_possible
			#### Adjust all frequencies (counts) by multiplying them with noun_ratios
			for k,v in all_possible.items():
				if '-' in k:
					all_possible[k] = v
				else:
					all_possible[k] = v * noun_ratio[k]
			all_possible = sorted(all_possible.items(), key=operator.itemgetter(1), reverse = True)
			# print '   ', all_possible
			if len(all_possible) == 1:
				# print '      ', all_possible[0][0]
				new_row.append(all_possible[0][0])
			elif all_possible[0][0] + 's' == all_possible[1][0]:
				if all_possible[1][1] > 0 and float(all_possible[0][1]) / all_possible[1][1] < 5:
					# print '      ', all_possible[1][0]
					new_row.append(all_possible[1][0])
				else:
					# print '      ', all_possible[0][0]
					new_row.append(all_possible[0][0])
			else:
				# print '      ', all_possible[0][0]
				new_row.append(all_possible[0][0])
		stemming_results[int(idx)] = list(set(new_row))


	### -----------------------
	### Write output file again
	### -----------------------
	output = open(output_name,'w')
	writer = csv.writer(output, quoting = csv.QUOTE_ALL)
	writer.writerow(['id','tags'])

	keylist = stemming_results.keys()
	keylist.sort()
	for key in keylist:
		writer.writerow([key,' '.join(stemming_results[key])])
	output.close()
	file_index += 1
	file_name = '../clusters/num' + str(file_index) + '.csv'
