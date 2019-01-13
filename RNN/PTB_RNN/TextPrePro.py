import os
import sys
import codecs


RAW_DATA = 'DataSet/ptb.test.txt'
VOCAB = 'ptb.vocab'
OUTPUT_DATA = 'ptb.test'


# 从文件中读取数据并编号
with codecs.open(VOCAB, 'r', 'utf-8') as f:
	vocab = [w.strip() for w in f.readlines()]
words2id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}


def get_id(word):
	return words2id[word] if word in words2id else words2id['<unk>']


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


