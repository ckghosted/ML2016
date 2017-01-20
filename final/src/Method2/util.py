from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter
from ast import literal_eval
from scipy import sparse
import numpy as np
import random
import string
import csv
import re
import gc

def prob_bool(p):
   assert p >= 0 and p <= 1, "Probability required to be between 0 and 1 !"
   return random.uniform(0, 1) <= p

def clean_html(text):
   return [re.sub('<.*?>', '', x) for x in text]

def clean_pos(data):
   p = pos_tag(data)
   cond = lambda x: x[1] == 'JJ' or x[1] == 'NN' or x[1] == 'NNS'
   data = [data[i] for i, x in enumerate(p) if cond(x)]
   return data

def get_token(sent, dic=None):
   sent = sent.decode('utf-8')
   lmtzr = WordNetLemmatizer()
   reg = re.compile(r'[^a-zA-Z\\]?')
   cond = lambda x: x.strip() and '\\' not in x
   data = [x for x in re.split(reg, sent) if cond(x)]
   cond = lambda x: len(x) > 3 and len(x) <= 20 #or x == '-'
   data = [x.lower() for x in data if not x.isdigit()]
   data = clean_pos(data)
   data = [x for x in data if x not in text.ENGLISH_STOP_WORDS]
   d = []
   for x in data:
      if cond(x):
         lemmed = lmtzr.lemmatize(x)
         d += [lemmed]
         if dic is not None:
            if lemmed in dic:
               if x[-2:] != 'al':
                  if dic[lemmed][-1][-2:] == 'al':
                     del dic[lemmed][-1]
                  dic[lemmed] += [x]
            else:
               dic[lemmed] = [x]
   data = d
   data = [x for x in data if cond(x)]
   if dic is not None:
      return data, dic
   else:
      return data

def get_token_origin(sent):
   sent = sent.decode('utf-8')
   reg = re.compile(r'[^a-zA-Z\\]?')
   cond = lambda x: x.strip() and '\\' not in x
   data = [x for x in re.split(reg, sent) if cond(x)]
   data = [x.lower() for x in data if not x.isdigit()]
   return data

def list_data(data):
   data = data.content.values.tolist()
   data = clean_html(data)
   data = [re.sub(r'\n', ' ', x) for x in data]
   return data   

def get_counter(data):
   big_str = ''
   for x in data:
      big_str += x
   data = big_str
   reg = re.compile(r'[^a-zA-Z]?')
   cond = lambda x: len(x) > 3 and len(x) <= 20 and not x.isdigit()
   data = [x.lower() for x in re.split(reg, big_str) if cond(x)]
   data = [x for x in data if x not in text.ENGLISH_STOP_WORDS]
   return Counter(data)

def get_diff_s(counter, word):
   if counter[word] + counter[word + 's'] == 0:
      return 0.0
   return float(counter[word] - counter[word + 's']) / (counter[word] + counter[word + 's'])

def get_tags(pdframe):
   data = pdframe.tags.values.tolist()
   data = list(set([x for d in data for x in d.split() if x]))
   return data

def get_tags_s(pdframe):
   data = get_tags(pdframe)
   data_s = set([x for x in data if (x[-1] == 's' and x[-2] != 's')])
   data_n = list(set(data) - data_s)[:len(data_s)]
   data_n = [(x, 0) for x in data_n]
   data_s = list(data_s)
   data_s = [(x, 1) for x in data_s]
   data = data_s + data_n
   random.shuffle(data)
   ret_tag = []
   ret_label = []
   for a, b in data:
      ret_tag += [a]
      ret_label += [b]
   return ret_tag, ret_label

def get_tag_and_probs(i):
   with open('../cluster_tags/tag' + str(i), 'r') as txt:
      tags, bi_tags, tri_tags = literal_eval(txt.readlines()[0])
   with open('../cluster_probs/prob' + str(i), 'r') as txt:
      probs, bi_probs = literal_eval(txt.readlines()[0])
   return dict(zip(tags, probs)), dict(zip(bi_tags, bi_probs)), tri_tags

def cluster_data(pdframe, nb_cluster):
   data = list_data(pdframe)
   data = [" ".join(get_token(x)) for x in data]
   vectorizer = TfidfVectorizer()
   features = vectorizer.fit_transform(data)
   gc.collect()
   u, s, v = sparse.linalg.svds(features, 200)
   n = Normalizer(copy=False)
   gc.collect()
   feat = n.fit_transform(u * s.transpose())
   km = KMeans(n_clusters=nb_cluster, n_init=1, init='k-means++', max_iter=100, verbose=2)
   print("Clustering by KMeans")
   km.fit(feat)
   print("Done!") 
   for x in range(nb_cluster):
      pdframe.iloc[np.where(km.labels_ == x)[0].tolist()].to_csv('../clusters/num' + str(x) + '.csv', index=False, quoting=csv.QUOTE_ALL)

def get_algo_symbol(pdframe):
   data = list_data(pdframe)
   def get_sym(sent):
      sent = sent.decode('utf-8')
      reg = re.compile(r'[^a-zA-Z\\]?')
      data = [x for x in re.split(reg, sent) if x.strip()]
      data = [x.lower() for x in data if x[0] == '\\' and len(x) >= 3]
      return data
   data = [get_sym(x) for x in data]
   data = [x for xx in data for x in xx]
   data = Counter(data)
   data = [k[1:] for k, v in data.iteritems() if v > 500]
   data = [x.split('\\')[0] if '\\' in x else x for x in data ]
   data = list(set(data))
   txt = open('../feat/algo', 'w')
   print >> txt, data
   txt.close()

def get_algo_set():
   txt = open('../feat/algo', 'r')
   return literal_eval(txt.readlines()[0])

if __name__ == '__main__':
   import pandas as pd
   x = pd.read_csv('../data/test.csv')
   get_algo_symbol(x)
