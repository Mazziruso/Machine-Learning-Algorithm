import tensorflow as tf
import os
import numpy as np
import codecs
import sys


# RAW_DATA = 'data/train.txt.en'
# VOCAB = 'data/en.vocab'
# OUTPUT_DATA = 'data/en.train'
RAW_DATA = 'data/train.txt.zh'
VOCAB = 'data/zh.vocab'
OUTPUT_DATA = 'data/zh.train'

with codecs.open(VOCAB, 'r', 'utf-8') as f:
	word2id = {}
	cnt = 0
	for w in f:
		word2id[w.strip()] = cnt
		cnt += 1


def get_id(word):
	return word2id[word] if word in word2id else word2id['<unk>']


fin = codecs.open(RAW_DATA, 'r', 'utf-8')
fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
try:
	for line in fin:
		words = line.strip().split() + ['<eos>']
		out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
		fout.write(out_line)
finally:
	fin.close()
	fout.close()
