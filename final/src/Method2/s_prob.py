from util import list_data, get_counter, get_tags, get_tags_s, get_diff_s
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from ast import literal_eval
import pandas as pd
import numpy as np
import os

def get_all_tags_s():
	all_tags = []
	for path in datas:
		data = pd.read_csv(path)
		tags = get_tags(data)
		tags = [w for w in tags if len(w) >= 3]
		tags = [w for w in tags if (w[-1] == 's' and w[-2] != 's')]
		all_tags += tags
	return all_tags

def get_test(x_test, x_test_bi):
	data = pd.read_csv('../data/test.csv')
	data = list_data(data)
	counter = get_counter(data)
	x_test = [get_diff_s(counter, x) for x in x_test]
	x_test_bi = [get_diff_s(counter, y) for x, y in x_test_bi]
	return x_test, x_test_bi

def gen_prob():
	index = 0
	tag_path = '../cluster_tags/tag' + str(index)
	while os.path.isfile(tag_path):
		print tag_path
		txt = open(tag_path, 'r')
		x_test_raw, x_test_raw_bi, _ = literal_eval(txt.readlines()[0])
		txt.close()
		prob, prob_bi = get_test(x_test_raw, x_test_raw_bi)
		for idx, x in enumerate(prob):
			if x > 0.7:
				prob[idx] = 1.0
			elif x <= 0.0:
				prob[idx] = 0.0
		for idx, x in enumerate(prob_bi):
			if x > 0.7:
				prob_bi[idx] = 1.0
			elif x <= 0.0:
				prob_bi[idx] = 0.0
		txt = open('../cluster_probs/prob' + str(index), 'w')
		print >> txt, (prob, prob_bi)
		txt.close()
		index += 1
		tag_path = '../cluster_tags/tag' + str(index)
if __name__ == '__main__':
	gen_prob()
