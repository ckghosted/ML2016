import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2,l1l2

## For autoencoder
from keras.layers import Input, Dense
from keras.models import Model

import os
import sys

# os.environ["THEANO_FLAGS"] = "device=gpu0"

data_path = sys.argv[1]
output_name = sys.argv[2]

# ================
# Load HW4 dataset
# ================
## Read title file
f_title = open(data_path+'title_StackOverflow.txt')
corpus = list()
for line in f_title:
    corpus.append(line.strip())

f_title.close()

## Read check_index file
f_check_index = open(data_path+'check_index.csv')
check_index = {}
## skip head:
line = f_check_index.readline()
for line in f_check_index:
    t = line.strip().split(',')
    check_index[int(t[0])] = [int(t[1]), int(t[2])]

f_check_index.close()

# ==================
# Feature Extraction
# ==================
import re
## Tokenize string
split_regex = r'[^a-zA-Z_]'
stopwords = set([u'using', u'use', u'file', u'files', u'get', u'way',
	             u'all', u'just', u'being', u'over', u'both', u'through',
	             u'yourselves', u'its', u'before', u'with', u'had', u'should',
	             u'to', u'only', u'under', u'ours', u'has', u'do', u'them',
	             u'his', u'very', u'they', u'not', u'during', u'now', u'him',
	             u'nor', u'did', u'these', u't', u'each', u'where', u'because',
	             u'doing', u'theirs', u'some', u'are', u'our', u'ourselves',
	             u'out', u'what', u'for', u'below', u'does', u'above',
	             u'between', u'she', u'be', u'we', u'after', u'here',
	             u'hers', u'by', u'on', u'about', u'of', u'against', u's',
	             u'or', u'own', u'into', u'yourself', u'down', u'your',
	             u'from', u'her', u'whom', u'there', u'been', u'few',
	             u'too', u'themselves', u'was', u'until', u'more',
	             u'himself', u'that', u'but', u'off', u'herself',
	             u'than', u'those', u'he', u'me', u'myself', u'this',
	             u'up', u'will', u'while', u'can', u'were', u'my', u'and',
	             u'then', u'is', u'in', u'am', u'it', u'an', u'as',
	             u'itself', u'at', u'have', u'further', u'their', u'if',
	             u'again', u'no', u'when', u'same', u'any', u'how',
	             u'other', u'which', u'you', u'who', u'most', u'such',
	             u'why', u'a', u'don', u'i', u'having', u'so', u'the',
	             u'yours', u'once'])

def tokenize(string):
    return [token for token in re.split(split_regex, string.lower()) \
            if token and (not token in stopwords)]

## Tokenize all titles
corpus_tok = [tokenize(s) for s in corpus]

## Implement a TF function
def tf(tokens):
    dict_TF = {}
    total_tokens = 0
    for token in tokens:
        dict_TF[token] = dict_TF.get(token, 0) + 1
        total_tokens += 1
    for key in dict_TF:
        dict_TF[key] /= float(total_tokens)
    return dict_TF

## Don't use simple bag of words or "inverse" document frequency, use TF-DF!
## Implement an DFs function
def dfs(corpus_tokens):
    N = len(corpus_tokens)
    ### Find unique token for each title:
    ### [['a', 'b', 'a'], ['b', 'c', 'd']] ==> [['a', 'b'], ['b', 'c', 'd']]
    uniqueTokens = [list(set(lst)) for lst in corpus_tokens]
    ### Flatten: [['a', 'b'], ['b', 'c', 'd']] ==> ['a', 'b', 'b', 'c', 'd']
    uniqueTokens = [token for tokens in uniqueTokens for token in tokens]
    ### Dict: ['a', 'b', 'b', 'c', 'd'] ==> {'a':1, 'b':2, 'c':1, 'd':1}
    uniqueTokens_dict = {}
    for token in uniqueTokens:
        uniqueTokens_dict[token] = uniqueTokens_dict.get(token, 0) + 1
    ### Take (log-scaled) document frequency directly, since a tag has very high
    ### probability to be included in the title
    return {k:np.log(v) for k, v in uniqueTokens_dict.iteritems()}

corpus_dfs = dfs(corpus_tok)

## Implement a TF-DF function
def tfdf(tokens, dfs):
    tfs = tf(tokens)
    tfDfDict = {}
    for token in tfs:
        tfDfDict[token] = tfs[token] * dfs[token]
    return tfDfDict

## Compute the TF-DF for all tokens in all titles
corpus_tfdf = [tfdf(t, corpus_dfs) for t in corpus_tok]

## Remove words that appear in less than 4 titles
token_freq_all = {}
for tklist in corpus_tok:
    for tk in tklist:
        token_freq_all[tk] = token_freq_all.get(tk, 0) + 1

corpus_tok_compact = []
for tklist in corpus_tok:
    tk_temp = []
    for tk in tklist:
        if token_freq_all[tk] > 3:
            tk_temp.append(tk)
    corpus_tok_compact.append(tk_temp)

corpus_dfs_compact = dfs(corpus_tok_compact)
# print len(corpus_dfs_compact)
corpus_tfdf_compact = [tfdf(t, corpus_dfs_compact) for t in corpus_tok_compact]

## Make feature matrix
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
x = vec.fit_transform(corpus_tfdf_compact).toarray()

# ===========
# Autoencoder
# ===========
## Reference: My code to train the 3162-1024-512-1024-3162 autoencoder
'''
input_word = Input(shape=(len(corpus_dfs_compact),))
encoded = Dense(1024, activation='relu')(input_word)
encoded = Dense(512, activation='relu')(encoded)
encoder = Model(input=input_word, output=encoded)
decoded = Dense(1024, activation='relu')(encoded)
decoded = Dense(len(corpus_dfs_compact), activation='sigmoid')(decoded)
autoencoder = Model(input=input_word, output=decoded)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
trnLoss = []
for i in xrange(4):
    hist = autoencoder.fit(x, x,
                           nb_epoch=25,
                           batch_size=128,
                           shuffle=True)
    encoder.save('encoders_20161201_compact3_2layers/encoder_%dep' % ((i+1)*25))
    trnLoss_tmp = hist.history.get('loss')
    trnLoss = trnLoss + trnLoss_tmp
'''
## 
from keras.models import load_model
encoder_load = load_model('./encoder')
encoded_x = encoder_load.predict(x)

# =======
# K-Means
# =======
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 40, random_state = 1002).fit(encoded_x)

result = {}
for k,v in check_index.iteritems():
    if kmeans.labels_[v[0]] == kmeans.labels_[v[1]]:
        result[k] = 1
    else:
        result[k] = 0

fout = open(output_name, 'w')
fout.write('ID,Ans\n')
for k,v in result.iteritems():
    fout.write('%d,%d\n' % (k,v))

fout.close()