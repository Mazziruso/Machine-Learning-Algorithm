import os
import collections
from collections import Counter
import numpy as np
import tensorflow as tf
import random


# Global Param
TRAIN_DATA = 'ptb.train'
VALID_DATA = 'ptb.valid'
TEST_DATA = 'ptb.test'
HIDDEN_SIZE = 300
NUM_LAYER = 2
VOCAB_SIZE = 10000
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35
VALID_BATCH_SIZE = 1
VALID_NUM_STEP = 1
NUM_EPOCH = 5
LSTM_KEEP_PROB = 0.9
EMBEDDING_KEEP_PROB = 0.9
MAX_GRAD_NORM = 5  # 梯度截断上限
SHARE_EMB_AND_SOFTMAX = True  # embedding层与softmax层是否参数共享


class PTBModel(object):
	def __init__(self, is_training, batch_size, num_steps):
		self.batch_size = batch_size
		self.num_steps = num_steps

		with tf.name_scope("INPUT"):
			self.inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, num_steps])
			self.labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, num_steps])

		with tf.name_scope("LSTM_CELL"):
			lstm_keep_p = LSTM_KEEP_PROB if is_training else 1.0
			lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=lstm_keep_p) for _ in range(NUM_LAYER)]
			cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

		with tf.name_scope("CELL_INITIAL"):
			self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)

		with tf.name_scope("EMBEDDING_LAYER"):
			embedding = tf.get_variable(name='embedding', shape=[VOCAB_SIZE, HIDDEN_SIZE])

		with tf.name_scope("EMBED_OUT"):
			cell_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

		# with tf.name_scope("LSTM_CELL_OUT"):
		# 	state = self.initial_state
		# 	if is_training:
		# 		cell_inputs = tf.nn.dropout(cell_inputs, EMBEDDING_KEEP_PROB)
		# 	cell_outputs, state = tf.nn.dynamic_rnn(cell, cell_inputs, initial_state=state, time_major=False)
		# 	cell_output = tf.reshape(tf.concat(cell_outputs, 1), shape=[-1, HIDDEN_SIZE])
		if is_training:
			cell_inputs = tf.nn.dropout(cell_inputs, EMBEDDING_KEEP_PROB)
		state = self.initial_state
		cell_output = []
		with tf.variable_scope("LSTM_CELL_OUT"):
			for time_step in range(num_steps):
				# 变量没有重用
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				outputs, state = cell(cell_inputs[:, time_step, :], state)
				cell_output.append(outputs)
		# 将输出reshape成[batch_size*num_steps, hidden_size]
		cell_output = tf.reshape(tf.concat(cell_output, 1), shape=[-1, HIDDEN_SIZE])

		with tf.name_scope("SOFTMAX_LAYER"):
			if SHARE_EMB_AND_SOFTMAX:
				weights = tf.transpose(embedding)
			else:
				weights = tf.get_variable("output_weights", shape=[HIDDEN_SIZE, VOCAB_SIZE])
			biases = tf.get_variable("output_biases", shape=[VOCAB_SIZE])
			# logits: shape=[batch_size*num_steps, vocab_size]
			logits = tf.nn.bias_add(tf.matmul(cell_output, weights), biases)

		with tf.name_scope("LOSS"):
			self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.labels, shape=[-1]), logits=logits)) / batch_size
			self.final_state = state

		# 只在训练时定义反向传播和梯度下降算法
		if not is_training:
			return
		else:
			trainable_variables = tf.trainable_variables()
			# 梯度截断，防止梯度爆炸出现loss divergence现象
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), MAX_GRAD_NORM)
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
			self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


# 使用给定的model在数据data上运行train_op并返回在全部数据上的交叉熵损失值 log perplexity
def run_epoch(session, model, batches, train_op, output_log, step):
	total_cost = 0.0
	iters = 0
	state = session.run(model.initial_state)
	# train an epoch
	for x, y in batches:
		cost, state, _ = session.run([model.loss, model.final_state, train_op], feed_dict={model.inputs: x, model.labels: y, model.initial_state: state})
		total_cost += cost
		iters += model.num_steps

		# display
		if output_log and step % 100 == 0:
			print("After %d steps, perplexity is %.3f" % (step, np.exp(total_cost/iters)))

		step += 1
	return step, np.exp(total_cost/iters)


# Read
def read_data(file_path):
	with open(file_path, 'r') as f:
		id_string = ' '.join([line.strip() for line in f.readlines()])
	id_list = [int(w) for w in id_string.split()]
	return id_list


# Batching
def make_batches(id_list, batch_size, num_steps):
	# 计算总的batch数
	num_batches = (len(id_list)-1) // (batch_size * num_steps)

	# 将数据整理为[batch_size, num_batches*num_steps]的二维数组
	data = np.array(id_list[:num_batches*batch_size*num_steps])
	data = np.reshape(data, [batch_size, -1])
	# 沿第二维度将数据切分成num_batches个batch,存入一个数组
	data_batched = np.split(data, num_batches, axis=1)

	# 对labels重复上述操作，但是每个位置向右移动一位，这里得到的是RNN每一步输出所需要预测的下一个单词
	labels = np.array(id_list[1:(num_batches*batch_size*num_steps+1)])
	labels = np.reshape(labels, [batch_size, -1])
	labels_batches = np.split(labels, num_batches, axis=1)

	return list(zip(data_batched, labels_batches))


if __name__ == '__main__':
	# 定义初始化函数
	initializer = tf.random_uniform_initializer(-0.05, 0.05)

	with tf.variable_scope("Language_model", reuse=None, initializer=initializer):
		train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

	with tf.variable_scope("Language_model", reuse=True, initializer=initializer):
		valid_model = PTBModel(False, VALID_BATCH_SIZE, VALID_NUM_STEP)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
		valid_batches = make_batches(read_data(VALID_DATA), VALID_BATCH_SIZE, VALID_NUM_STEP)
		test_batches = make_batches(read_data(TEST_DATA), VALID_BATCH_SIZE, VALID_NUM_STEP)

		step = 0
		for i in range(NUM_EPOCH):
			step, train_pplx = run_epoch(sess, train_model, train_batches, train_model.train_op, True, step)
			_, valid_pplx = run_epoch(sess, valid_model, valid_batches, tf.no_op(), False, 0)
			print("Epoch: %d, Train Perplexity: %.3f, Valid Perplexity: %.3f" % (i+1, train_pplx, valid_pplx))

		_, test_pplx = run_epoch(sess, valid_model, test_batches, tf.no_op(), False, 0)
		print("Train Done! Test Perplexity: %.3f" % test_pplx)


