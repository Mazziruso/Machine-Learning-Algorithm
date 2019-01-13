import os
import numpy as np
import collections
from collections import Counter
import codecs
from operator import itemgetter


RAW_DATA = 'DataSet/ptb.train.txt'
VOCAB_OUTPUT = 'ptb.vocab'

counter = Counter()
with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
	for line in f:
		for word in line.strip().split():
			counter[word] += 1

# 按词频对单词进行排序
sorted_word2cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word2cnt]

sorted_words.append('<eos>')

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as f:
	for word in sorted_words:
		f.write(word + '\n')
