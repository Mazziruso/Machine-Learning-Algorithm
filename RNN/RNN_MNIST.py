import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# mnist data
mnist = input_data.read_data_sets('data', one_hot=True)

# LOG DIR

logdir = "LOG_RNN/"

# global constant/parameter
# learning_rate = 1E-3
# batch_size = 128
epochs = 25
n_inputs = 28
n_steps = 28
# n_hidden_units = 128
n_classes = 10
num_tests = 512


def summary_variables(var):
	with tf.name_scope("summaries"):
		with tf.name_scope("mean"):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
		with tf.name_scope("stddev"):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
			tf.summary.scalar('stddev', stddev)
		with tf.name_scope("max"):
			tf.summary.scalar('max', tf.reduce_max(var))
		with tf.name_scope("min"):
			tf.summary.scalar('min', tf.reduce_min(var))
		with tf.name_scope("histogram"):
			tf.summary.histogram('histogram', var)


# rnn architecture
def rnn(x, weights, biases, batch_size, n_hidden_units):
	with tf.name_scope("Input_Reshape"):
		x = tf.reshape(x, [-1, n_inputs])

	with tf.name_scope("Input_Layer"):
		with tf.name_scope("weights"):
			summary_variables(weights['in'])
		with tf.name_scope("biases"):
			summary_variables(biases['out'])
		x_in = tf.nn.bias_add(tf.matmul(x, weights['in']), biases['in'])
		x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])

	with tf.name_scope("LSTM_CELL"):
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_units, forget_bias=1.0, state_is_tuple=True)
		init_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

	with tf.name_scope("Hidden_Layer"):
		outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)
		with tf.name_scope("Final_State"):
			summary_variables(final_state[1])

	with tf.name_scope("Output_Layer"):
		with tf.name_scope("weights"):
			summary_variables(weights['out'])
		with tf.name_scope("biases"):
			summary_variables(biases['out'])
		result = tf.nn.bias_add(tf.matmul(final_state[1], weights['out']), biases['out'])  #
	return result


def model(learning_rate, batch_size, n_hidden_units, hparam):
	tf.reset_default_graph()

	x = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs], name='input_images')
	y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='input_labels')

	with tf.name_scope("Input_Data"):
		tf.summary.image('images', tf.reshape(x, [-1, 28, 28, 1]), 8)

	weights = {
		'in': tf.Variable(tf.random_normal(shape=[n_inputs, n_hidden_units]), dtype=tf.float32),
		'out': tf.Variable(tf.random_normal(shape=[n_hidden_units, n_classes]), dtype=tf.float32)
	}
	biases = {
		'in': tf.Variable(tf.random_normal(shape=[n_hidden_units]), dtype=tf.float32),
		'out': tf.Variable(tf.random_normal(shape=[n_classes]), dtype=tf.float32)
	}

	with tf.name_scope("Predict"):
		pred = rnn(x, weights, biases, batch_size, n_hidden_units)
	with tf.name_scope("Loss"):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
		tf.summary.scalar('Loss', loss)
	with tf.name_scope("Train"):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	with tf.name_scope("Accuracy"):
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
		tf.summary.scalar('accuracy', accuracy)

	with tf.Session() as sess:
		#
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(logdir=logdir+'cp_' + hparam, graph=sess.graph)
		sess.run(tf.global_variables_initializer())
		total_batch = int(mnist.train.num_examples/batch_size)
		cnt = 0
		for epoch in range(epochs):
			for step in range(total_batch):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
				sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
				if cnt % 20 == 0:
					summary = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
					writer.add_summary(summary, cnt*batch_size)
				cnt += 1

			batch_x, batch_y = mnist.test.next_batch(batch_size)
			batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
			test_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			print("epoch: %d, accuracy: %.5f" % (epoch, test_acc))


def make_hparam_string(hparam):
	return "learning_rate_%d_batch_size_%d_n_hidden_units_%d" % (hparam[0]*10000, hparam[1], hparam[2])


if __name__ == '__main__':
	batch_size = 128
	for n_hidden_unit in [32, 64, 128, 256]:
		for learning_rate in [1E-2, 1E-3, 1E-4]:
			hparam = make_hparam_string([learning_rate, batch_size, n_hidden_unit])
			print("Starting run for %s" % hparam)
			model(learning_rate, batch_size, n_hidden_unit, hparam)
	print("Train Done!")
