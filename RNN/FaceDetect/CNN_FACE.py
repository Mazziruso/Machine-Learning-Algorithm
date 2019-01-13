import os
import sys
import read_data
import numpy as np
from PIL import Image
import tensorflow as tf


# data set
faces = read_data.read_data_sets(one_hot=True, reshape=False)

# LOG DIR
log_dir = 'LOG_CNN/'

# global parameter
# learning_rate = 1E-3
# batch_size = 40
epochs = 200
display_iter = 10


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


# conv layer
def conv2d(input_data, weight, bias, name):
	with tf.name_scope("conv2d"):
		res_conv = tf.nn.conv2d(input_data, weight, strides=[1, 1, 1, 1], padding='SAME', name=name)
		res_conv = tf.nn.bias_add(res_conv, bias)
	with tf.name_scope("activate"):
		return tf.nn.relu(res_conv)


# pool layer
def pool2d(input_data, name, k=2):
	with tf.name_scope("pool2d"):
		return tf.nn.max_pool(input_data, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def norm2d(l_input, name, lsize=4):
	with tf.name_scope("norm2d"):
		return tf.nn.lrn(l_input, depth_radius=lsize/2, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)


def drop2d(input_data, name, keep_prob):
	with tf.name_scope("drop2d"):
		return tf.nn.dropout(input_data, keep_prob=keep_prob, name=name)


# Net Structure
def cnn_net(x, weights, biases, p_conv, p_hidden):
	# layer1
	with tf.name_scope("convolution_layer1"):
		conv1 = conv2d(x, weights['w1'], biases['b1'], 'conv1')
		pool1 = pool2d(conv1, 'pool1', k=2)
		drop1 = drop2d(pool1, 'drop1', p_conv)
	# layer2
	with tf.name_scope("convolution_layer2"):
		conv2 = conv2d(drop1, weights['w2'], biases['b2'], 'conv2')
		pool2 = pool2d(conv2, 'pool2', k=2)
		drop2 = drop2d(pool2, 'drop2', p_conv)
	# layer3
	with tf.name_scope("convolution_layer3"):
		conv3 = conv2d(drop2, weights['w3'], biases['b3'], 'conv3')
		pool3 = pool2d(conv3, 'pool3', k=2)
		drop3 = drop2d(pool3, 'drop3', p_conv)
	# fc layer1
	with tf.name_scope("full_connection_layer1"):
		fc_in = tf.reshape(drop3, shape=[-1, weights['wd'].get_shape().as_list()[0]])
		with tf.name_scope("fc1"):
			with tf.name_scope("weights"):
				summary_variables(weights['wd'])
			with tf.name_scope("biases"):
				summary_variables(biases['bd'])
			with tf.name_scope("Wx_plus_b"):
				fc_out1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc_in, weights['wd']), biases['bd']), name='activation')
			with tf.name_scope("dropout"):
				fc_drop1 = tf.nn.dropout(fc_out1, keep_prob=p_hidden, name='fc1_dropout')
	# output layer
	with tf.name_scope("output_layer"):
		with tf.name_scope("output"):
			with tf.name_scope("weights"):
				summary_variables(weights['out'])
			with tf.name_scope("biases"):
				summary_variables(biases['out'])
			with tf.name_scope("Wx_plus_b"):
				out = tf.nn.bias_add(tf.matmul(fc_drop1, weights['out']), biases['out'])
				tf.summary.histogram('output_Py', out)
	return out


# model
def model(learning_rate, batch_size, p_conv, p_hidden, hparam):
	#
	tf.reset_default_graph()
	# 输入占位符
	X = tf.placeholder(dtype=tf.float32, shape=[None, 57, 47, 1], name='input_images')
	Y = tf.placeholder(dtype=tf.float32, shape=[None, 40], name='input_labels')
	p_keep_conv = tf.placeholder(dtype=tf.float32, name='conv_keep_prob')
	p_keep_hidden = tf.placeholder(dtype=tf.float32, name='hidden_keep_prob')

	# show and summary the image
	with tf.name_scope('input_data'):
		tf.summary.image('images', X, 8)


	# weights and biases
	weights = {
		'w1': tf.Variable(tf.random_normal(shape=[3, 3, 1, 20], stddev=0.01), dtype=tf.float32),
		'w2': tf.Variable(tf.random_normal(shape=[3, 3, 20, 40], stddev=0.01), dtype=tf.float32),
		'w3': tf.Variable(tf.random_normal(shape=[6, 6, 40, 80], stddev=0.01), dtype=tf.float32),
		'wd': tf.Variable(tf.random_normal(shape=[80*8*6, 960], stddev=0.01), dtype=tf.float32),
		'out': tf.Variable(tf.random_normal(shape=[960, 40], stddev=0.01), dtype=tf.float32)
	}

	biases = {
		'b1': tf.Variable(tf.random_normal(shape=[20], stddev=0.01), dtype=tf.float32),
		'b2': tf.Variable(tf.random_normal(shape=[40], stddev=0.01), dtype=tf.float32),
		'b3': tf.Variable(tf.random_normal(shape=[80], stddev=0.01), dtype=tf.float32),
		'bd': tf.Variable(tf.random_normal(shape=[960], stddev=0.01), dtype=tf.float32),
		'out': tf.Variable(tf.random_normal(shape=[40], stddev=0.01), dtype=tf.float32),
	}

	# cost and accuracy
	pred = cnn_net(X, weights, biases, p_keep_conv, p_keep_hidden)
	with tf.name_scope("LOSS"):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
		tf.summary.scalar('LOSS', cost)
	with tf.name_scope("TRAIN"):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	with tf.name_scope("ACCURACY"):
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1)), tf.float32))
		tf.summary.scalar('ACCURACY', accuracy)

	# session
	with tf.Session() as sess:
		# merge all summary
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(log_dir+'cp_'+hparam, sess.graph)
		# initialization
		sess.run(tf.global_variables_initializer())
		# iteration
		total_batch = int(faces.train.num_examples/batch_size)
		cnt = 0
		for epoch in range(epochs):
			for step in range(total_batch):
				batch_x, batch_y = faces.train.next_batch(batch_size)
				sess.run(optimizer, feed_dict={X: batch_x, Y:batch_y, p_keep_conv:p_conv, p_keep_hidden:p_hidden})
				if cnt % display_iter == 0:
					summary = sess.run(merged, feed_dict={X: batch_x, Y:batch_y, p_keep_conv: 1.0, p_keep_hidden: 1.0})
					writer.add_summary(summary, cnt*batch_size)
				cnt += 1

				validation_x, validation_y = faces.validation.next_batch(batch_size)
				valid_acc = sess.run(accuracy, feed_dict={X: validation_x, Y: validation_y, p_keep_conv: 1.0, p_keep_hidden: 1.0})
				print("epoch: %d, accuracy: %.5f" % (epoch, valid_acc))
		test_x, test_y = faces.test.next_batch(batch_size)
		test_acc = sess.run(accuracy, feed_dict={X: test_x, Y: test_y, p_keep_conv: 1.0, p_keep_hidden: 1.0})
		print("Finish Train")
		print("test accuracy: %.5f" % test_acc)


def make_hparam_string(hparam):
	return "lr_%.0E_bs_%d_conv_p_%.0E, hidden_p_%.0E" % (hparam[0], hparam[1], hparam[2], hparam[3])


if __name__ == '__main__':
	batch_size = 40
	learning_rate = 1E-3
	for p_keep_conv in [0.5, 0.75, 1.0]:
		for p_keep_hidden in [0.5, 0.75, 1.0]:
			hparam = make_hparam_string([learning_rate, batch_size, p_keep_conv, p_keep_hidden])
			print("Starting run for %s" % hparam)
			model(learning_rate, batch_size, p_keep_conv, p_keep_hidden, hparam)
	print("Done ALL!")
