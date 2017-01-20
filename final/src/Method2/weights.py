from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from gensim.models import Word2Vec
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.models import load_model
#import tensorflow as tf
from collections import Counter
from scipy import sparse
import numpy as np
import pandas as pd
import string
import logging
import sys #argv[1] is the path containing all the data
import re
import os.path
import random
import gc

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
embed_SIZE = 700

class process_features:

	def __init__(self, args):
		#paths
		assert len(args) == 2, "Unmatched size of arguments, should be \'data path\', \'content name\' "
		#data
		self.data_path = args[0]
		#self.data = self.load_data()
		#model
		self.vectorizer = []
		self.val_vectorizer = []
		self.feat_path = "../feat/" + args[1]
		#tags
		self.tagset = []
		self.val_tagset = []
#		self.get_tags()
		self.maxtag = 20.0
		self.doclen = 0

	def get_test(self):
		print "Getting physics..."
		data = pd.read_csv( self.data_path + "/test.csv")
		data = data.content.values.tolist()
		data = self.clean_html(data)
		temp = []
		data = [re.sub(r'\n', ' ', x) for x in data]
		for d in data:
			if self.get_token(d):
				temp += [self.get_token(d)]
		mul = 10
		x_test = [x for sublist in temp for x in sublist]
		c = Counter(x_test)
		x_test = list(set(x_test))
		x_test = [x for x in x_test if c[x] > 25]
		ll = lambda x: float(len(x))
		lt = map(ll, temp)
		x_test = [[doc.count(w) * mul / lt[i] for i, doc in enumerate(temp)] for w in x_test]
		x_test = np.array(x_test)
		#x_train = np.concatenate((x_train, np.zeros((x_train.shape[0], self.doclen - x_train.shape[1]))), axis=1)
		gc.collect()
		print "Doing LSA"
		print "SVD...."
		u, s, v = sparse.linalg.svds(x_test, embed_SIZE)
		n = Normalizer(copy=False)
		x_test = n.fit_transform(u * s.transpose())
		return x_test
	
	def load_data(self):
		print "Loading data..."
		#topics = ['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel']
		topics = ['biology', 'crypto', 'robotics']
		data = []
		for x in topics:
			data.append( pd.read_csv( self.data_path + "/" + x + ".csv") )
		print "Done!"
		return data

	def gen_train(self, x_train_c, val):
		mul = 10
		x_train = [x for sublist in x_train_c for x in sublist]
		c = Counter(x_train)
		x_train = list(set(x_train))
		x_train = [x for x in x_train if c[x] > 25]
		print "Producing targets"
		y_train = []
		weights = []
		if not val:
			xx = self.tagset
			self.doclen = len(x_train_c)
		else:
			xx = self.val_tagset
			copy = x_train
		for x in x_train:
			if x in xx:
				weights += [xx[x] / self.maxtag if xx[x] >= self.maxtag else 1.0]
				y_train += [1]
			else:
				weights += [0.7]
				y_train += [0]
		weights = np.array(weights)
		print "Embedding training words..."
		ll = lambda x: float(len(x))
		lt = map(ll, x_train_c)
		print len(lt)
		x_train = [[doc.count(w) * mul / lt[i] for i, doc in enumerate(x_train_c)] for w in x_train]
		x_train = np.array(x_train)
		if val :
			assert self.doclen != 0, "Error!"
			x_train = np.concatenate((x_train, np.zeros((x_train.shape[0], self.doclen - x_train.shape[1]))), axis=1)
		gc.collect()
		print "Doing LSA"
		print "SVD...."
		u, s, v = sparse.linalg.svds(x_train, embed_SIZE)
		n = Normalizer(copy=False)
		x_train = n.fit_transform(u * s.transpose())
		y_train = np.array(y_train)
		print x_train
		if not val:
			return x_train, y_train, weights
		else:
			return x_train, y_train, weights, copy

	def get_train(self):
		if os.path.isfile("../feat/feats.npy"):
			print "Feature already exists, Loading in..."
			x_train, x_val, y_train, y_val, t_weights, v_weights, tags = np.load("../feat/feats.npy")
		else:
			(x_train_c, x_val_c) = self.get_content(self.feat_path)
			x_train, y_train, t_weights = self.gen_train(x_train_c, False)
			x_val, y_val, v_weights, tags = self.gen_train(x_val_c, True)
			#rearranging dims
			print "Rearranging dims of training sets"
			x_train = self.rearrange(x_train, x_val)
			#apply mean_shift
			x_train -= np.mean(x_train, axis=0)
			x_val -= np.mean(x_val, axis=0)
			print "Saving feats"
			np.save("../feat/feats", (x_train, x_val, y_train, y_val, t_weights, v_weights, tags))
		return x_train, x_val, y_train, y_val, t_weights, v_weights, tags

	########### Helper Functions ###########

	def clean_html(self, text):
		return [re.sub('<.*?>', '', x) for x in text]

	def get_token(self, sent):
		sent = sent.decode('utf-8')
		lmtzr = WordNetLemmatizer()
		#doing spliting and keeping '-' i.e. ['ribosome', 'binding', '-', 'sites', 'translation', 'synthetic', '-', 'biology']
		#reg = re.compile(r'[^a-zA-Z0-9-]?')
		#punc_set = string.punctuation.replace("-", "")
		#data = [x.strip() for x in re.split(reg, sent) if (x.strip() and (x.strip() not in punc_set))]
		reg = re.compile(r'[^a-zA-Z]?')
		data = [x for x in re.split(reg, sent) if x.strip()]
		cond = lambda x: len(x) > 3 and len(x) <= 20 #or x == '-'
		data = [x.lower() for x in data if not x.isdigit()]
		data = self.clean_pos(data)
		data = [x for x in data if x not in text.ENGLISH_STOP_WORDS]
		data = [lmtzr.lemmatize(x) for x in data if cond(x)]
#		data = [lmtzr.lemmatize(x, pos='v') for x in data if cond(x)] #not sure
		data = [x for x in data if cond(x)]
		return data

	def get_tags(self):
		datas = self.data
		print "Getting tags..."
		tags, val_tags = [], []
		for idx, x in enumerate(datas):
			data = x.tags.values.tolist()
			if idx != (len(datas) - 1):
				for x in data:
					tags += self.get_token(x)
			else:
				for x in data:
					val_tags += self.get_token(x)
		self.tagset = Counter(tags)
		self.val_tagset = Counter(val_tags)

	def clean_pos(self, data):
		p = pos_tag(data)
		cond = lambda x: x[1] == 'JJ' or x[1] == 'NN' #or x[1] == ':'
		data = [data[i] for i, x in enumerate(p) if cond(x)]
		return data

	def get_content(self, path):
		datas = self.data
		print "Parsing data..."
		if os.path.isfile(path):
			print "Feature already exists, Loading in..."
			all_datas, val_data = np.load(path)
		else:
			all_datas = []
			for idx, x in enumerate(datas):
				data = x.content.values.tolist()
				data = self.clean_html(data)
				temp = []
				data = [re.sub(r'\n', ' ', x) for x in data]
				for d in data:
					if self.get_token(d):
						temp += [self.get_token(d)]
				print idx, '/', len(datas) - 1
				if idx != (len(datas) - 1):
					all_datas += temp
				else:
					val_data = temp
			print "Saving feats"
			np.save(path, (all_datas, val_data))
		print "Done!"
		return (all_datas, val_data)

	def rearrange(self, x, y):
		n = Normalizer(copy=False)
		v1_trunc = n.fit_transform(x[:y.shape[0]].transpose())
		v1 = x.transpose()
		v2 = n.fit_transform(y.transpose())
		def swap(v, a, b):
			temp = np.copy(v[a])
			v[a] = v[b]
			v[b] = temp
		for idx, x in enumerate(v2):
			_dots = np.dot(v1_trunc[idx:], x)
			_result = idx + np.argmax(_dots)
			swap(v1_trunc, idx, _result)
			swap(v1, idx, _result)
		return v1.transpose()

feat = process_features(sys.argv[1:])
x_train, x_val, y_train, y_val, t_weights, v_weights, tags = feat.get_train()
tags = np.array(tags)

best_score = 0
for x in range(500):
	model = Sequential()
	model.add(Dense(256, input_shape=(embed_SIZE,), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.4))
	model.add(BatchNormalization())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.4))
	model.add(BatchNormalization())
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.5))
	model.add(BatchNormalization())
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',\
		optimizer=Adam(lr=0.001),\
		metrics=['accuracy'])

	model.fit(x_train, y_train,\
		batch_size=128,\
		nb_epoch=100,\
		validation_data=(x_val, y_val, v_weights),\
		callbacks=[EarlyStopping(patience=3)],\
		sample_weight=t_weights)

	n_large = 50
	p = model.predict(x_val, batch_size=64)
	p = np.array([x[0] for x in p])
	out = np.argpartition(p, -1 * n_large)[-1 * n_large:]
	out = out[np.argsort(p[out])[::-1]]
	t = tags[out]
	print t
	np.set_printoptions(threshold=np.nan)
	print feat.val_tagset
	s = 0 
	a = []
	for x in t:
		if x in feat.val_tagset:
			s += feat.val_tagset[x]
			a += [(x, feat.val_tagset[x])]
	print a
	print "Score : ", s
	if s > best_score:
		best_score = s
		model.save('../model/best.h5')
		txt = open("monitor.log", 'w')
		print>>txt, a
		txt.close()

print "loading Model..."
model = load_model('../model/best.h5')
print "Getting test..."
x_test = feat.get_test()
n_large = 50
p = model.predict(x_test, batch_size=64)
p = np.array([x[0] for x in p])
out = np.argpartition(p, -1 * n_large)[-1 * n_large:]
out = out[np.argsort(p[out])[::-1]]
t = tags[out]
print t


def myf1(y_true, y_pred):
	return fbeta_score(y_true, y_pred, beta=1)

#TODO
#POS tagging keeps only adj, noun.
#then change the tagets to one hot encoding of output words(think that with the split of '-', the OOV will be rare)
#apply another NN for classfication of where '-' should involve in
#This will not be end-to-end.
#Q : better approach?

#TODO
#first filter out words that maybe tags(regression one by one).
#use the reduced set of tags to train...
