import os
import numpy as np
import tensorflow as tf
import codecs
import string


ENGLISH_DATA = 'data/train.tags.en-zh.en'
CHINESE_DATA = 'data/train.tags.en-zh.zh'
OUTPUT_EN = 'data/train.txt.en'
OUTPUT_CH = 'data/train.txt.zh'

if __name__ == '__main__':

	# fin = codecs.open(ENGLISH_DATA, 'r', 'utf-8')
	# fout = codecs.open(OUTPUT_EN, 'w', 'utf-8')
	# try:
	# 	for line in fin:
	# 		line = line.lower().strip()
	# 		if not (line.startswith('<') and line.endswith('>')):
	# 			for c in string.punctuation:
	# 				line = line.replace(c, ' %c ' % c)
	# 			fout.write(line + '\n')
	# finally:
	# 	fin.close()
	# 	fout.close()

	fin = codecs.open(CHINESE_DATA, 'r', 'utf-8')
	fout = codecs.open(OUTPUT_CH, 'w', 'utf-8')
	try:
		for line in fin:
			line = line.strip()
			if not (line.startswith('<') and line.endswith('>')):
				line = ' '.join(line.strip()) + '\n'
				fout.write(line)
	finally:
		fin.close()
		fout.close()


