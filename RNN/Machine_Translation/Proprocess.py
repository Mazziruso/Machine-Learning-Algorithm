import tensorflow as tf
import numpy as numpy
import random
import os
from collections import Counter
import codecs
from operator import itemgetter


# RAW_DATA = 'data/train.txt.en'
# VOCAB_OUTPUT = 'data/en.vocab'
RAW_DATA = 'data/train.txt.zh'
VOCAB_OUTPUT = 'data/zh.vocab'

counter = Counter()
with codecs.open(RAW_DATA, 'r', 'utf=8') as f:
	for line in f:
		for word in line.strip().split():
			counter[word] += 1

# sorted counter in freq
sorted_word2cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)  # 返回元组列表
sorted_words = [w[0] for w in sorted_word2cnt]

sorted_words = ['<unk>', '<sos>', '<eos>'] + sorted_words
if len(sorted_words) > 4000:
	sorted_words = sorted_words[:4000]

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as f:
	for word in sorted_words:
		f.write(word + '\n')

