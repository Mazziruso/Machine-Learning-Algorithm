import sys
import os
import numpy as np
import tensorflow as tf
from collections import Counter
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# global param
save_dir = "checkpoints/text8.ckpt"
vocab2int = {}
int2vocab = {}


# 构造skip-gram模型的上下文
def get_target(words, idx, window_size=5):
	'''
	返回下标idx单词的上下文列表
	:param words: 词汇表
	:param idx: 中心单词下标
	:param window_size: 窗口大小
	:return:
	'''
	target_window = window_size
	# 检测边界
	start = idx - target_window if (idx - target_window) > 0 else 0
	end = idx + target_window + 1 if (idx + target_window + 1) < len(words) else len(words)
	# 输出上下文单词列表
	target_words = set(words[start:idx] + words[idx+1:end])
	return list(target_words)


# 构造batch
def next_batch(words, batch_size, window_size=5):
	'''
	返回一个获取batch的生成器
	:param words: 文本的词汇表
	:param batch_size: batch大小
	:param window_size: skip-gram窗口大小
	:return:
	'''
	# 仅取整数batch，循环构造
	words_len = len(words)
	num_batch = words_len // batch_size
	num_batch += 1
	words.extend(words[0:(num_batch*batch_size-words_len)])
	words_len = len(words)

	for idx in range(0,words_len,batch_size):
		x, y = [], []
		batch = words[idx:idx+batch_size]
		for i in range(batch_size):
			batch_x = batch[i]
			batch_y = get_target(batch, i, window_size)
			# x与yshape统一
			x.extend([batch_x] * len(batch_y))
			y.extend(batch_y)
		yield x, y


# 定义函数完成文本预处理
def word_pre_process(text, freq=5):
	'''
	文本预处理
	:param text: 文本数据
	:param freq: 词频阈值
	:return: 单词列表
	'''
	text = text.lower()
	text.replace('.', ' <PERIOD> ')
	text.replace(',', ' <COMMA> ')
	text.replace('"', ' <QUOTATION_MARK> ')
	text.replace(';', ' <SEMICOLON> ')
	text.replace('!', ' <EXCLAMATION_MARK> ')
	text.replace('?', ' <QUESTION_MARK> ')
	text.replace('(', ' <LEFT_PAREN> ')
	text.replace(')', ' <RIGHT_PAREN> ')
	text.replace('--', ' <HYPHENS> ')
	text.replace(':', ' <COLON> ')
	# text.replace('\n', ' <NEW_LINE> ')

	words = text.split()
	# 剔除低频词，减少噪音影响
	word_count = Counter(words)
	trimmed_words = [word for word in words if word_count[word] > freq]
	return trimmed_words


# 高频词采样
def sampled(int_words, threshold):
	'''
	对高频词进行采样，加快训练（softmax的计算）
	:param int_words: 词汇表（数字化后）
	:param threshold: 剔除概率阈值
	:return: 采样后的词汇列表
	'''
	t = 1E-5
	# 统计单词出现频次
	int_word_counts = Counter(int_words)
	total_count = len(int_words)
	# 计算单词频率
	word_freqs = {w: c/total_count for w, c in int_word_counts.items()}
	# 根据公式计算被删除概率
	drop_prob = {w: 1-np.sqrt(t/word_freqs[w]) for w in int_word_counts}
	# 采样
	sampled_words = [word for word in int_words if drop_prob[word] < threshold]
	return sampled_words


# 构造词汇表
def vocalTable(file_dir):
	global vocab2int
	global int2vocab

	with open(file_dir) as f:
		text = f.read()
	# 文本预处理并分词
	words = word_pre_process(text)
	vocab = set(words)
	# 构建映射表
	vocab2int = {w: c for c, w in enumerate(vocab)}
	int2vocab = {c: w for c, w in enumerate(vocab)}
	# 将原文本装换到int列表
	int_text = [vocab2int[w] for w in words]
	# 高频采样
	train_words = sampled(int_text, threshold=0.6)
	return train_words


# 不是one-hot向量表示，而是单个值来表示，因此输入向量形状为(batch_size,)，labels形状为(batch_size,1)
# Input Layer, Embedding Layer, Negative Layer
def netron_net(train_words, epochs, batch_size, window_size, learning_rate, embedding_size=200):
	vocab_size = len(train_words)
	train_graph = tf.Graph()
	num_sampled = 100  # 每个batch随机选择的类别数（输出层子集S的大小）
	with train_graph.as_default():
		# input layer
		inputs = tf.placeholder(dtype=tf.int32, shape=[None], name='inputs')
		labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name='labels')
		# embedding layer
		# 权重矩阵
		embedding = tf.Variable(tf.random_normal(shape=[vocab_size, embedding_size], stddev=0.1), dtype=tf.float32)
		# 实现lookup，模拟one-hot输入向量与权重矩阵相乘，输出低维词向量
		embed = tf.nn.embedding_lookup(embedding, inputs)
		# negative sampling layer
		# 负采样解决计算输出层softmax时计算量过大的问题(one-hot形式下)，这一层包含输出层权重矩阵计算，直接输出负采样后的loss(交叉熵函数)
		softmax_w = tf.Variable(tf.random_normal([vocab_size, embedding_size], dtype=tf.float32))
		softmax_b = tf.Variable(tf.random_normal([vocab_size], dtype=tf.float32))
		loss = tf.nn.sampled_softmax_loss(weights=softmax_w, biases=softmax_b, labels=labels, inputs=embed, num_sampled=100, num_classes=vocab_size)
		# 作平均后才是损失函数
		cost = tf.reduce_mean(loss)
		# 优化操作
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	# 随机验证并计算余弦相似度
	with train_graph.as_default():
		# 随机选取一些单词（非ont-hot表示）
		valid_size = 16
		valid_window = 100
		# 从不同位置各选8个单词
		valid_examples = random.sample(range(valid_window), valid_size//2)
		valid_examples.extend(random.sample(range(1000, 1000+valid_window), valid_size//2))
		valid_size = len(valid_examples)
		# 构造验证单词集为tf.constant类型
		valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
		# 每个词向量进行归一化（embedding中每一行）
		normalized_embedding = tf.divide(embedding, tf.sqrt(tf.reduce_sum(tf.square(embedding), axis=1, keep_dims=True)))
		# 验证单词的词向量
		valid_embed = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
		# 计算余弦相似度
		similarity = tf.matmul(valid_embed, tf.transpose(normalized_embedding))

	# Session开启
	with train_graph.as_default():
		saver = tf.train.Saver()
	with tf.Session(graph=train_graph) as sess:
		iter = 1
		total_loss = 0
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
			for X, Y in next_batch(train_words, batch_size, window_size):
				loss, _ = sess.run([cost, optimizer], feed_dict={inputs: X, labels: np.array(Y)[:, None]})
				total_loss += loss
				if iter%10 == 0:
					print("Epoch %d/%d" % (epoch+1, epochs))
					print("Avg Train Loss: %.4f" % total_loss)
					total_loss = 0
				# 计算与验证集单词中相似的词
				if iter%1000 == 0:
					sim = sess.run(similarity, feed_dict={inputs: X, labels: np.array(Y)[:, None]})
					for i in range(valid_size):
						valid_word = int2vocab[valid_examples[i]]
						top_k = 8  # 取最相似的8个
						nearest = (-sim[i, :]).argsort()[1:top_k+1]
						log = "Nearest to [%s]:" % valid_word
						for k in range(top_k):
							log = "%s %s," % (log, int2vocab[nearest[k]])
						print(log)
				iter += 1
		saver.save(sess, save_dir)
		return sess.run(normalized_embedding)


if __name__ == '__main__':

	text_dir = 'text8'
	# 预处理文本并获取训练词汇表
	train_words = vocalTable(text_dir)

	# 模型训练
	embed_mat = netron_net(train_words, epochs=10, batch_size=1000, window_size=10, learning_rate=0.001)
	# t-sne可视化
	vis_words = 500  # 降维可视化单词数
	tsne = TSNE()
	embed_tsne = tsne.fit_transform(embed_mat[:vis_words, :])
	fig, ax = plt.subplots(nrows=14, ncols=14)
	for idx in range(vis_words):
		plt.scatter(*embed_tsne[idx, :])
		plt.annotate(int2vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

