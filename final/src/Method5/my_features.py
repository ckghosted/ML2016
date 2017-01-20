import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import re
import csv
from nltk.stem.porter import PorterStemmer
import sys

## corpus_file: 'test.csv' (for physics), 'biology.csv', 'cooking.csv', etc.
corpus_file = sys.argv[1]
## Read the corpus
corpus = pd.read_csv(corpus_file, encoding = 'utf-8')

## pos_file: the file path of the POS-tagging results for the considered corpus_file
## (e.g., 'pos_dict.csv' (for physics), 'biology_pos_dict.csv', 'cooking_pos_dict.csv', etc.
pos_file = sys.argv[2]
## Read pos_tag results as dictionary
pos_fh = open(pos_file)
reader = csv.DictReader(pos_fh)
pos_dict = {}
for idx,row in enumerate(reader):
	pos = row['POS'].split()
	pos_dict[row['word']] = {pos_count.split(':')[0]:int(pos_count.split(':')[1]) for pos_count in pos}

pos_fh.close()
### Calculate the noun-ratio for each word
### (the proportion of time this word being tagged as a noun)
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

## Keep hyphen
punct = set(string.punctuation)
punct.remove('-')

my_stopwords = set(stopwords.words('english'))

def remove_punct(word_list):
	return ''.join(ch for ch in word_list if ch not in punct)

def remove_html(context):
	cleaner = re.compile('<.*?>')
	clean_text = re.sub(cleaner,'',context)
	return clean_text

def remove_newline(context):
	cleaner = re.compile('\n+')
	clean_text = re.sub(cleaner,' ',context)
	return clean_text

# ===================
# Real tag statistics
if not corpus_file == 'test.csv':
	all_tags = {}
	for i in xrange(corpus.shape[0]):
		tags = corpus.iloc[i,3].split(' ')
		for tag in tags:
			all_tags[tag] = all_tags.get(tag, 0) + 1

if False:
	## Any upper letter?
	for tag in all_tags:
		for ch in tag:
			if ch.isupper():
				print tag, 'has upper'
				break
	## ==> not any upper letter in tags

	## Any non-alphabet?
	for tag in all_tags:
		cleaner = re.compile('-')
		if not re.sub(cleaner,'',tag).isalpha():
			print tag, 'is not purely alphabet'
	## ==> 'ccp4', 'cas9', '3d-structure', and 't7-promoter' are all valud tags

	## End with 's'?
	count = 0
	for tag in all_tags:
		if tag[-1] == 's':
			count += 1

	print count, 'out of', len(all_tags), 'end with "s"'

	## End with 'ing'?
	count = 0
	for tag in all_tags:
		if tag[-3:] == 'ing':
			count += 1

	print count, 'out of', len(all_tags), 'end with "ing"'

	## How many tags with noun_ratio < 0.5? What do they look like?
	for tag in all_tags:
		if tag in noun_ratio and noun_ratio[tag] < 0.5:
			print tag, noun_ratio[tag]

# (Real tag statistics end)
# =========================


## From http://stackoverflow.com/questions/323750/how-to-access-previous-next-element-while-for-looping
## neighborhood: Get the previous and the next elements while looping a list
##     Example: for a list = ['A', 'B', 'C', 'D']
##     Generate: (None, 'A', 'B'), ('A', 'B', 'C'), ('B', 'C', 'D'), and ('C', 'D', None)
def neighborhood(iterable):
    iterator = iter(iterable)
    prev_item = None
    current_item = next(iterator)  # throws StopIteration if empty.
    for next_item in iterator:
        yield (prev_item, current_item, next_item)
        prev_item = current_item
        current_item = next_item
    yield (prev_item, current_item, None)
## Usage:
## >>> lst = ['A', 'B', 'C', 'D']
## >>> for pre,cur,nxt in neighborhood(lst):
## >>>     print pre,cur,nxt

## neighborhood2: Extend neighborhood() to previous 2 and next 2 words
##     Example: for a list = ['A', 'B', 'C', 'D']
##     Generate: (None, None, 'A', 'B', 'C'), (None, 'A', 'B', 'C', 'D'),
##               ('A', B', 'C', 'D', None), and ('B', C', 'D', None, None)
def neighborhood2(iterable):
    iterator = iter(iterable)
    prev_prev_prev_item = None
    prev_prev_item = None
    prev_item = None
    current_item = next(iterator)  # throws StopIteration if empty.
    for next_item in iterator:
    	# skip the first iteration (None, None, None, [0], [1])
    	if not prev_item is None:
        	yield (prev_prev_prev_item, prev_prev_item, prev_item, current_item, next_item)
        prev_prev_prev_item = prev_prev_item
        prev_prev_item = prev_item
        prev_item = current_item
        current_item = next_item
    yield (prev_prev_prev_item, prev_prev_item, prev_item, current_item, None)
    # Add one more iteration ([-3], [-2], [-1], None, None)
    yield (prev_prev_item, prev_item, current_item, None, None)

## basic_statistics: 
##     pre_pre, pre, nxt, and nxt_nxt: stem of the previous 2 and next 2 words
##     tk: a token in title or content
##     tk_type: 'title' or 'content'
def basic_statistics(pre_pre, pre, tk, nxt, nxt_nxt, tk_type):
	global capital_grams
	global all_lower_grams
	global all_upper_grams
	global before_ques
	global times_in_each_title
	global times_in_each_content
	## Check if the tk
	##     (1) Begins with a capital letter (e.g., 'Abc'),
	##     (2) All uppercased (e.g. 'ABC')
	##     (3) Else (e.g., 'abc', 'aBc', 'abC', or 'aBC')
	if tk[0].isupper():
		capital_grams[tk.lower()] = capital_grams.get(tk.lower(), 0) + 1
		all_upper = True
		for ch in tk:
			if not ch.isupper():
				all_upper = False
				break
		if all_upper:
			all_upper_grams[tk.lower()] = all_upper_grams.get(tk.lower(), 0) + 1
	else:
		all_lower_grams[tk.lower()] = all_lower_grams.get(tk.lower(), 0) + 1
	## After consider uppercase and lowercase, transform all into lower case
	tk = tk.lower()
	if tk_type == 'title':
		## Here, tk must in title, just update times_in_each_title and positions_in_each_title
		## title_tokens: all tokens in the considered title (no stopwords)
		if tk in times_in_each_title:
			times_in_each_title[tk][idx] = title.count(tk)
		else:
			times_in_each_title[tk] = {idx:title.count(tk)}
		positions = [float(i+1) / len(title_tokens) for i, x in enumerate(title_tokens) if x.lower() == tk]
		if tk in positions_in_each_title:
			positions_in_each_title[tk][idx] = np.mean(positions)
		else:
			positions_in_each_title[tk] = {idx:np.mean(positions)}
	elif tk_type == 'content':
		## Here, tk must in content, just update times_in_each_content and positions_in_each_content
		## content_tokens: all tokens in the considered content (no stopwords)
		if tk in times_in_each_content:
			times_in_each_content[tk][idx] = content.count(tk)
		else:
			times_in_each_content[tk] = {idx:content.count(tk)}
		positions = [float(i+1) / len(content_tokens) for i, x in enumerate(content_tokens) if x.lower() == tk]
		if tk in positions_in_each_content:
			positions_in_each_content[tk][idx] = np.mean(positions)
		else:
			positions_in_each_content[tk] = {idx:np.mean(positions)}
	## Check weather or not the (stemmed) neighbors contain the following keywords:
	## 'tag', 'understand', 'studi', 'introduct', 'explain', 'principl', 'interpret',
	## 'differ', 'book', and 'knowledg'
	if pre_pre == kw_tag or pre == kw_tag or nxt == kw_tag or nxt_nxt == kw_tag:
		around_kw_tag[tk] = around_kw_tag.get(tk, 0) + 1
	if pre_pre == kw_understand or pre == kw_understand or nxt == kw_understand or nxt_nxt == kw_understand:
		around_kw_understand[tk] = around_kw_understand.get(tk, 0) + 1
	if pre_pre == kw_study or pre == kw_study or nxt == kw_study or nxt_nxt == kw_study:
		around_kw_study[tk] = around_kw_study.get(tk, 0) + 1
	if pre_pre == kw_introduction or pre == kw_introduction or nxt == kw_introduction or nxt_nxt == kw_introduction:
		around_kw_introduction[tk] = around_kw_introduction.get(tk, 0) + 1
	if pre_pre == kw_explain or pre == kw_explain or nxt == kw_explain or nxt_nxt == kw_explain:
		around_kw_explain[tk] = around_kw_explain.get(tk, 0) + 1
	if pre_pre == kw_principle or pre == kw_principle or nxt == kw_principle or nxt_nxt == kw_principle:
		around_kw_principle[tk] = around_kw_principle.get(tk, 0) + 1
	if pre_pre == kw_interpret or pre == kw_interpret or nxt == kw_interpret or nxt_nxt == kw_interpret:
		around_kw_interpret[tk] = around_kw_interpret.get(tk, 0) + 1
	if pre_pre == kw_difference or pre == kw_difference or nxt == kw_difference or nxt_nxt == kw_difference:
		around_kw_difference[tk] = around_kw_difference.get(tk, 0) + 1
	if pre_pre == kw_book or pre == kw_book or nxt == kw_book or nxt_nxt == kw_book:
		around_kw_book[tk] = around_kw_book.get(tk, 0) + 1
	if pre_pre == kw_knowledge or pre == kw_knowledge or nxt == kw_knowledge or nxt_nxt == kw_knowledge:
		around_kw_knowledge[tk] = around_kw_knowledge.get(tk, 0) + 1

## Temp data structure for basic statistics
capital_grams = {} # {word:number}
all_lower_grams = {} # {word:number}
all_upper_grams = {} # {word:number}
pattern_ques = re.compile("\s*([a-z0-9]+)\?") # 0 or more spaces + alphabet (or digit) + '?'
before_ques = {} # {word:number}
stemmer1 = PorterStemmer()
kw_tag = stemmer1.stem('tag')
around_kw_tag = {} # {word:number}
kw_understand = stemmer1.stem('understand')
around_kw_understand = {} # {word:number}
kw_study = stemmer1.stem('study')
around_kw_study = {} # {word:number}
kw_introduction = stemmer1.stem('introduction')
around_kw_introduction = {} # {word:number}
kw_explain = stemmer1.stem('explain')
around_kw_explain = {} # {word:number}
kw_principle = stemmer1.stem('principle')
around_kw_principle = {} # {word:number}
kw_interpret = stemmer1.stem('interpret')
around_kw_interpret = {} # {word:number}
kw_difference = stemmer1.stem('difference')
around_kw_difference = {} # {word:number}
kw_book = stemmer1.stem('book')
around_kw_book = {} # {word:number}
kw_knowledge = stemmer1.stem('knowledge')
around_kw_knowledge = {} # {word:number}
times_in_title_first = {} # {word:number}
times_in_title_last = {} # {word:number}
times_in_content_first = {} # {word:number}
times_in_content_last = {} # {word:number}
times_in_each_title = {} # {word:{id:number}}
times_in_each_content = {} # {word:{id:number}}
positions_in_each_title = {} # {word:{id:positions}}
positions_in_each_content = {} # {word:{id:positions}}

## Debug: check the ratio of being around some keywords
# count = 0
# total = 0
# for tk in around_kw_explain:
# 	total += 1
# 	if tk in all_tags:
# 		count += 1
# print float(count) / total

## ----------------------------------------------------------------------
## Loop all the documents (questions) to produce all the basic statistics
## ----------------------------------------------------------------------
for i in xrange(corpus.shape[0]):
	idx = corpus.iloc[i,0]
	## Preprocessing
	title = remove_punct(corpus.iloc[i,1]).strip()
	content = remove_punct(remove_newline(remove_html(corpus.iloc[i,2]))).strip()
	## Replace en-dash into hyphen
	# title = re.sub('\xe2\x80\x93', '-', title)
	# content = re.sub('\xe2\x80\x93', '-', content)
	title = re.sub(u'\u2013', '-', title)
	content = re.sub(u'\u2013', '-', content)
	## Tokenization (used in later "for" loops)
	title_tokens = [tk for tk in title.split() if not (tk.lower() in my_stopwords or len(tk) < 3 or (not tk[0].isalpha()))]
	content_tokens = [tk for tk in content.split() if not (tk.lower() in my_stopwords or len(tk) < 3 or (not tk[0].isalpha()))]
	## [NOTE] Keep uppercase in 'title_tokens' and 'content_tokens',
	##        but transform 'title' and 'content' into all lowercase
	title = title.lower()
	content = content.lower()
	## Consider previous 2 and next 2 words (transfer into stem first)
	title_tokens_stem = {tk:stemmer1.stem(tk) for tk in title_tokens}
	title_tokens_stem[None] = None
	for pre_pre,pre,tk,nxt,nxt_nxt in neighborhood2(title_tokens):
		if tk is None:
			continue
		else:
			basic_statistics(title_tokens_stem[pre_pre], title_tokens_stem[pre], tk, title_tokens_stem[nxt], title_tokens_stem[nxt_nxt], 'title')
	content_tokens_stem = {tk:stemmer1.stem(tk) for tk in content_tokens}
	content_tokens_stem[None] = None
	for pre_pre,pre,tk,nxt,nxt_nxt in neighborhood2(content_tokens):
		if tk is None:
			continue
		else:
			basic_statistics(content_tokens_stem[pre_pre], content_tokens_stem[pre], tk, content_tokens_stem[nxt], content_tokens_stem[nxt_nxt], 'content')
	## [NOTE] Transform 'title_tokens' and 'content_tokens' into all lowercase
	title_tokens = [tk.lower() for tk in title_tokens]
	content_tokens = [tk.lower() for tk in content_tokens]
	## Check if word being the first or the last in title or content
	if len(title_tokens) > 0:
		times_in_title_first[title_tokens[0]] = times_in_title_first.get(title_tokens[0], 0) + 1
		times_in_title_last[title_tokens[-1]] = times_in_title_last.get(title_tokens[-1], 0) + 1
		times_in_content_first[content_tokens[0]] = times_in_content_first.get(content_tokens[0], 0) + 1
		times_in_content_last[content_tokens[-1]] = times_in_content_last.get(content_tokens[-1], 0) + 1
	## Find all words before question mark
	title_before_ques = pattern_ques.findall(corpus.iloc[i,1].lower())
	content_before_ques = pattern_ques.findall(remove_newline(remove_html(corpus.iloc[i,2])))
	for tk in title_before_ques + content_before_ques:
		if not tk in my_stopwords:
			before_ques[tk] = before_ques.get(tk, 0) + 1

## Temp list for basic statistics
## (They will become columns in the resulting feature data frame)
words = []
n_occurence = []
n_title = []
n_question = []
ratio_capital = []
ratio_upper = []
ratio_before_ques = []
ratio_noun = []
ratio_around_kw_tag = []
ratio_around_kw_understand = []
ratio_around_kw_study = []
ratio_around_kw_introduction = []
ratio_around_kw_explain = []
ratio_around_kw_principle = []
ratio_around_kw_interpret = []
ratio_around_kw_difference = []
ratio_around_kw_book = []
ratio_around_kw_knowledge = []
mean_occur_in_title = []
mean_position_in_title = []
ratio_in_title_first = []
ratio_in_title_last = []
mean_occur_in_content = []
mean_position_in_content = []
ratio_in_content_first = []
ratio_in_content_last = []
mean_occur_in_question = []
## Only consider words (uni-grams) that we have calculated their noun_ratio in POS tagging
for word in noun_ratio:
	if not '-' in word:
		words.append(word)
		n_occurence.append(capital_grams.get(word,0) + all_lower_grams.get(word,0))
		n_title.append(len(times_in_each_title.get(word,{})))
		n_question.append(len(times_in_each_title.get(word,{})) + len(times_in_each_content.get(word,{})))
		ratio_capital.append(np.float64(capital_grams.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		ratio_upper.append(np.float64(all_upper_grams.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		ratio_before_ques.append(np.float64(before_ques.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		if word in noun_ratio:
			ratio_noun.append(noun_ratio[word])
		else:
			ratio_noun.append(float('nan'))
		ratio_around_kw_tag.append(np.float64(around_kw_tag.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		ratio_around_kw_understand.append(np.float64(around_kw_understand.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		ratio_around_kw_study.append(np.float64(around_kw_study.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		ratio_around_kw_introduction.append(np.float64(around_kw_introduction.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		ratio_around_kw_explain.append(np.float64(around_kw_explain.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		ratio_around_kw_principle.append(np.float64(around_kw_principle.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		ratio_around_kw_interpret.append(np.float64(around_kw_interpret.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		ratio_around_kw_difference.append(np.float64(around_kw_difference.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		ratio_around_kw_book.append(np.float64(around_kw_book.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		ratio_around_kw_knowledge.append(np.float64(around_kw_knowledge.get(word,0)) / (capital_grams.get(word,0) + all_lower_grams.get(word,0)))
		if word in times_in_each_title:
			mean_occur_in_title.append(np.mean([v for k,v in times_in_each_title[word].items()]))
			mean_position_in_title.append(np.mean([v for k,v in positions_in_each_title[word].items()]))
			ratio_in_title_first.append(np.float64(times_in_title_first.get(word,0)) / len(times_in_each_title[word]))
			ratio_in_title_last.append(np.float64(times_in_title_last.get(word,0)) / len(times_in_each_title[word]))
			if word in times_in_each_content:
				times_in_each_question = {k:(times_in_each_title[word].get(k,0)+times_in_each_content[word].get(k,0)) for k in set(times_in_each_title[word].keys() + times_in_each_content[word].keys())}
				mean_occur_in_question.append(np.mean([v for k,v in times_in_each_question.items()]))
			else:
				mean_occur_in_question.append(np.mean([v for k,v in times_in_each_title[word].items()]))
		else:
			mean_occur_in_title.append(0)
			mean_position_in_title.append(float('nan'))
			ratio_in_title_first.append(0)
			ratio_in_title_last.append(0)
			if word in times_in_each_content:
				mean_occur_in_question.append(np.mean([v for k,v in times_in_each_content[word].items()]))
			else:
				mean_occur_in_question.append(float('nan'))
		if word in times_in_each_content:
			mean_occur_in_content.append(np.mean([v for k,v in times_in_each_content[word].items()]))
			mean_position_in_content.append(np.mean([v for k,v in positions_in_each_content[word].items()]))
			ratio_in_content_first.append(np.float64(times_in_content_first.get(word,0)) / len(times_in_each_content[word]))
			ratio_in_content_last.append(np.float64(times_in_content_last.get(word,0)) / len(times_in_each_content[word]))
		else:
			mean_occur_in_content.append(0)
			mean_position_in_content.append(float('nan'))
			ratio_in_content_first.append(0)
			ratio_in_content_last.append(0)

is_end_with_s = []
is_end_with_ing = []
str_length = []
is_tag = []
for word in words:
	if word[-1] == 's':
		is_end_with_s.append(1)
		is_end_with_ing.append(0)
	elif word[-3:] == 'ing':
		is_end_with_ing.append(1)
		is_end_with_s.append(0)
	else:
		is_end_with_s.append(0)
		is_end_with_ing.append(0)
	str_length.append(len(word))
	## Add outcome (is a valid tag?) for corpus other than 'test.csv'
	if not corpus_file == 'test.csv':
		if word in all_tags:
			is_tag.append(1)
		else:
			is_tag.append(0)

out_file = corpus_file.split('.')[0] + '_df.csv' ## 'test.csv' ==> 'test_df.csv', etc.
with open(out_file, 'w+') as fh_out:
	table=[words, n_occurence, n_title, n_question, ratio_capital, ratio_upper, ratio_before_ques, ratio_noun, ratio_around_kw_tag, ratio_around_kw_understand, \
	       ratio_around_kw_study, ratio_around_kw_introduction, ratio_around_kw_explain, ratio_around_kw_principle, ratio_around_kw_interpret, ratio_around_kw_difference, \
	       ratio_around_kw_book, ratio_around_kw_knowledge, \
	       mean_occur_in_title, mean_position_in_title, ratio_in_title_first, ratio_in_title_last, \
	       mean_occur_in_content, mean_position_in_content, ratio_in_content_first, ratio_in_content_last, \
	       mean_occur_in_question, \
	       is_end_with_s, is_end_with_ing, str_length]
	if not corpus_file == 'test.csv':
		table.append(is_tag)
	df = pd.DataFrame(table)
	df = df.transpose()
	cols = ['word', 'n_occurence', 'n_title', 'n_question', 'ratio_capital', 'ratio_upper', 'ratio_before_ques', 'ratio_noun', 'ratio_around_kw_tag', 'ratio_around_kw_understand', \
	       'ratio_around_kw_study', 'ratio_around_kw_introduction', 'ratio_around_kw_explain', 'ratio_around_kw_principle', 'ratio_around_kw_interpret', 'ratio_around_kw_difference', \
	       'ratio_around_kw_book', 'ratio_around_kw_knowledge', \
	       'mean_occur_in_title', 'mean_position_in_title', 'ratio_in_title_first', 'ratio_in_title_last', \
	       'mean_occur_in_content', 'mean_position_in_content', 'ratio_in_content_first', 'ratio_in_content_last', \
	       'mean_occur_in_question', \
	       'is_end_with_s', 'is_end_with_ing', 'str_length']
	if not corpus_file == 'test.csv':
		cols.append('is_tag')
	df.columns = cols
	df.to_csv(fh_out, mode = 'a', index = False, header = True)
	fh_out.close()










