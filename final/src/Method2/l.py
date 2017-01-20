from util import get_token, get_token_origin, clean_html, list_data, get_counter, get_diff_s
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from keras.models import Sequential, Model
from keras.models import load_model
from collections import Counter
from scipy import sparse
import numpy as np
import pandas as pd
import string
import sys #argv[1] is the path containing all the data
import re
import os.path
import gc

embed_SIZE = 700
#add_zero_uni = set(['force'])
#add_zero_bi = set(['wave', 'vector', 'particle', 'field'])
#add_zero = add_zero_uni | add_zero_bi

#def plfy(word, mode):
#	if mode == 'uni':
#		if word in add_zero_uni:
#			return (word + 's')
#		else:
#			return word
#	elif mode == 'bi':
#		if word in add_zero_bi:
#			return (word + 's')
#		else:
#			return word
#	elif mode == 'all':
#		if word in add_zero_all:
#			return (word + 's')
#		else:
#			return word
#	else:
#		raise ValueError
#		return

#def modify_end_bigram(word):
#	def reduce_end(w):
#		if len(w) <= 3:
#			return w
#		ends = w[-2:]
#		if ends == 'al':
#			return None
#		elif ends == 'ic':
#			return w + 's'
#		else:
#			return w
#	word = plfy(w, 'bi')
#	word = reduce_end(word)
#	return word

def pad_to_embed(z, embed_size):
	if z.shape[0] >= embed_size:
		return z
	pad = np.zeros(z.shape[1])
	for x in range(embed_size - z.shape[0]):
		z = np.vstack((z, pad))
	return z

def push_dict(dic, w):
	if w in dic:
		dic[w] += 1
	else:
		dic[w] = 1

def get_bigram(x_test_c, raw_tags, doc_len):
	counter = {}
	max_freq_th = 400
	freq_th = doc_len / 10
	if freq_th > max_freq_th:
		freq_th = max_freq_th
	for doc in x_test_c:
		for i, w in enumerate(doc):
			if w in raw_tags:
				if i != 0:
					front = (doc[i-1], w)
					push_dict(counter, front)
				if i != len(doc) - 1:
					back = (w, doc[i+1])
					push_dict(counter, back)
	counter = Counter(counter)
	ret = [k for k, v in counter.iteritems() if v >= freq_th]
	gram_dict = Counter([x for xs in ret for x in xs])
	return ret, gram_dict

def get_trigram(x_test_c, raw_tags, doc_len):
	counter = {}
	freq_th = doc_len / 20
	max_freq_th = 400
	if freq_th > max_freq_th:
		freq_th = max_freq_th
	for doc in x_test_c:
		for i, w in enumerate(doc):
			if w in raw_tags:
				if i > 1:
					front = (doc[i-2], doc[i-1], w)
					push_dict(counter, front)
				if i < len(doc) - 2:
					back = (w, doc[i+1], doc[i+2])
					push_dict(counter, back)
				if i != 0 and i != len(doc) - 1:
					middle = (doc[i-1], w, doc[i+1])
					push_dict(counter, middle)
	counter = Counter(counter)
	return [k for k, v in counter.iteritems() if v >= freq_th]

def prep_data(data_path):
	print "Getting physics..."
	data = pd.read_csv(data_path)
	data = list_data(data)
	true_counter = get_counter(data)
	return data, true_counter

def get_test_origin(data):
	temp = [get_token_origin(d) for d in data]
	return [x for sub in temp for x in sub]

def get_test(data):
	temp = []
	dic = {}
	temp = []
	for d in data:
		a, dic = get_token(d, dic)
		if a:
			temp += [a]
	dic = { key:max(set(value), key=value.count) for key, value in dic.iteritems() }
	mul = 10
	x_test = [x for sublist in temp for x in sublist]
	c = Counter(x_test)
	most_set = [x[0] for x in c.most_common(20)]
	x_test = list(set(x_test))
	x_test = [x for x in x_test if c[x] >= 200]
	txt = open('../feat/test_feats', 'w')
	print >> txt, (temp, x_test, dic, most_set)
	return temp, x_test, dic, most_set

def process_test(x_test_c=None, x_test=None, x=None, feat_path=None):
	length = len(x_test)
	if feat_path is not None:
		return np.load(feat_path)
	else:
		assert x_test_c is not None, "Please specify x_test_c."
		assert x_test is not None, "Please specify x_test."
		assert x is not None, "Please specify iteration"
	ll = lambda x: float(len(x))
	lt = map(ll, x_test_c)
	mul = 10.0
	qq = 0
	for w in x_test:
		if not qq:
			z = np.array([doc.count(w) * mul / lt[i] for i, doc in enumerate(x_test_c)])
			qq = 1 
		else:
			z = np.vstack((z, np.array([doc.count(w) * mul / lt[i] for i, doc in enumerate(x_test_c)])))
	x_test = pad_to_embed(z, embed_SIZE + 100)
	gc.collect()
	print "Doing LSA"
	print "SVD...."
	u, s, v = sparse.linalg.svds(x_test, embed_SIZE)
	n = Normalizer(copy=False)
	x_test = n.fit_transform(u * s.transpose())
	np.save('../cluster_svds/svd' + str(x), x_test)
	return x_test[:length]

def clean_dup_bigram(bigram_tags, trigram_tags):
	any_same = lambda x, y: (x[0] == y[0] and x[1] == y[1]) \
								or (x[0] == y[1] and x[1] == y[2])
	for x in trigram_tags:
		for y in bigram_tags:
			if any_same(y, x):
				del y
	return bigram_tags

def get_physics_tags(data_path, x):
	print "loading Model..."
	model = load_model('../model/best.h5')
	print "Getting test..."
	data, true_counter = prep_data(data_path)
	x_test_c, tags, dic, most_set = get_test(data)
#	x_test_origin = get_test_origin(data)
	doc_len = len(x_test_c)
	x_test = process_test(x_test_c, tags, x)
	tags = np.array(tags)
	n_large = 40
	p = model.predict(x_test, batch_size=64)
	p = np.array([x[0] for x in p])
	if p.shape[0] < n_large:
		n_large = p.shape[0]
	out = np.argpartition(p, -1 * n_large)[-1 * n_large:]
	out = out[np.argsort(p[out])[::-1]]
	tags = tags[out].tolist()
	tags += most_set

	s_cond = lambda x: x[-1] == 's' and x[-2] != 's'
	bigram_tags, bidict = get_bigram(x_test_c, tags, doc_len)
	bigram_tags = [(dic[a], dic[b]) for a, b in bigram_tags if a != b and not s_cond(dic[a])]
	bigram_tags = list(set(bigram_tags))
	tags = [x for x in tags if x in bidict]
	add_tags = [k for k, v in bidict.iteritems() if v >= 4]
	tags += add_tags

	tags = [dic[t] for t in tags]
	tags = list(set(tags))
#	tags = [plfy(x, 'all') for x in tags if len(plfy(x, 'all')) > 4]
	trigram_tags = get_trigram(x_test_c, tags, doc_len)
	any_same = lambda a, b, c: len(set([a, b, c])) == 3
	trigram_tags = [(dic[a], dic[b], dic[c]) for a, b, c in trigram_tags if any_same(a, b, c)]
	trigram_tags = list(set(trigram_tags))
#	bigram_tags = [( a, modify_end_bigram(b) ) for a, b in bigram_tags if modify_end_bigram(b) is not None]
	print tags, bigram_tags, trigram_tags
	return tags, bigram_tags, trigram_tags

def out_cluster_tags():
	x = 0
	file_path = '../clusters/num' + str(x) + '.csv'
	while os.path.isfile(file_path):
		tags, bi_tags, tri_tags = get_physics_tags(file_path, x)
		txt = open('../cluster_tags/tag' + str(x), 'w')
		print >> txt, (tags, bi_tags, tri_tags)
		x += 1
		file_path = '../clusters/num' + str(x) + '.csv'

if __name__ == '__main__':
	out_cluster_tags()

#TODO
#POS tagging keeps only adj, noun.
#then change the tagets to one hot encoding of output words(think that with the split of '-', the OOV will be rare)
#apply another NN for classfication of where '-' should involve in
#This will not be end-to-end.
#Q : better approach?

#TODO
#first filter out words that maybe tags(regression one by one).
#use the reduced set of tags to train...
