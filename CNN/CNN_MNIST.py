import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# GLOBAL PARAM
# data set
mnist = input_data.read_data_sets('data', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape([-1, 28, 28, 1])
teX = teX.reshape([-1, 28, 28, 1])

# hyper parameters
decay = 0.9
# batch_size = 128
test_size = 256
train_iters = 25
display_iters = 10
log_dir = 'LOG/'


def conv2d(input_data, weight, bias, name):
	res = tf.nn.conv2d(input_data, weight, strides=[1, 1, 1, 1], padding='SAME', name=name)
	return tf.nn.relu(res)


def pool2d(input_data, name, k=2):
	return tf.nn.max_pool(input_data, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


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


def nn_layer_summary(input_tensor, weight, bias, layer_name, act_func=tf.nn.relu):
	with tf.name_scope(layer_name):
		with tf.name_scope("weights"):
			summary_variables(weight)
		with tf.name_scope("biases"):
			summary_variables(bias)
		with tf.name_scope("Wx_plus_b"):
			pre_act = tf.nn.bias_add(tf.matmul(input_tensor, weight), bias)
			tf.summary.histogram('pre_activations', pre_act)
			activations = act_func(pre_act, name='activation')
			tf.summary.histogram('activations', activations)
			return activations


def cnn(x, weights, biases, keep_prob_conv, keep_prob_hidden):
	# layer1
	with tf.name_scope("convolution_layer1"):
		conv1 = conv2d(x, weights['w1'], biases['b1'], name='conv1')
		pool1 = pool2d(conv1, name='pool1', k=2)
		drop1 = tf.nn.dropout(pool1, keep_prob=keep_prob_conv, name='drop1')

	# layer2
	with tf.name_scope("convolution_layer2"):
		conv2 = conv2d(drop1, weights['w2'], biases['b2'], 'conv2')
		pool2 = pool2d(conv2, 'pool2', k=2)
		drop2 = tf.nn.dropout(pool2, keep_prob=keep_prob_conv, name='drop2')

	# layer3
	with tf.name_scope("convolution_layer3"):
		conv3 = conv2d(drop2, weights['w3'], biases['b3'], 'conv3')
		pool3 = pool2d(conv3, name='pool3', k=2)
		drop3 = tf.nn.dropout(pool3, keep_prob=keep_prob_conv, name='drop3')

	# fc1
	with tf.name_scope("full_connection_layer1"):
		fc_in = tf.reshape(drop3, shape=[-1, weights['wd'].get_shape().as_list()[0]])
		fc1 = nn_layer_summary(fc_in, weights['wd'], biases['bd'], 'fc1', tf.nn.relu)
		drop_fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob_hidden)

	# output layer
	with tf.name_scope("output_layer"):
		with tf.name_scope("weights"):
			summary_variables(weights['out'])
		with tf.name_scope("biases"):
			summary_variables(biases['out'])
		with tf.name_scope("Wx_plus_b"):
			out = tf.nn.bias_add(tf.matmul(drop_fc1, weights['out']), biases['out'])
			tf.summary.histogram('output_Py', out)
	return out


def model(learning_rate, batch_size, dropout_conv, dropout_hidden, hparam):
	#
	tf.reset_default_graph()
	# 输入占位符
	X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input_images')
	Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='input_labels')
	p_keep_conv = tf.placeholder(dtype=tf.float32, name='conv_keep_prob')
	p_keep_hidden = tf.placeholder(dtype=tf.float32, name='hidden_keep_prob')

	# summarize images and labels
	with tf.name_scope("input_data"):
		tf.summary.image('images', X, 8)

	# weights and biases
	weights = {
		'w1': tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01), dtype=tf.float32),
		'w2': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01), dtype=tf.float32),
		'w3': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01), dtype=tf.float32),
		'wd': tf.Variable(tf.random_normal([128*4*4, 625], stddev=0.01), dtype=tf.float32),
		'out': tf.Variable(tf.random_normal([625, 10], stddev=0.01), dtype=tf.float32),
	}

	biases = {
		'b1': tf.Variable(tf.random_normal([32], stddev=0.01), dtype=tf.float32),
		'b2': tf.Variable(tf.random_normal([64], stddev=0.01), dtype=tf.float32),
		'b3': tf.Variable(tf.random_normal([128], stddev=0.01), dtype=tf.float32),
		'bd': tf.Variable(tf.random_normal([625], stddev=0.01), dtype=tf.float32),
		'out': tf.Variable(tf.random_normal([10], stddev=0.01), dtype=tf.float32),
	}

	# cost and accuracy
	pred = cnn(X, weights, biases, p_keep_conv, p_keep_hidden)
	with tf.name_scope("LOSS"):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
		tf.summary.scalar('LOSS', cost)
	with tf.name_scope("Train"):
		optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(cost)
	with tf.name_scope("ACCURACY"):
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1)), tf.float32))
		tf.summary.scalar('ACCURACY', accuracy)
	with tf.name_scope("Predict"):
		predict_op = tf.argmax(pred, axis=1)

	# session
	with tf.Session() as sess:
		# merge summaries
		merged = tf.summary.merge_all()
		# 写到指定磁盘路径中
		writer = tf.summary.FileWriter(log_dir+'cp_'+hparam, sess.graph)
		# initializer
		sess.run(tf.global_variables_initializer())
		# iterator train
		cnt = 0
		for i in range(train_iters):
			train_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size))
			for start, end in train_batch:
				sess.run(optimizer, feed_dict={X: trX[start:end], Y: trY[start:end],
												p_keep_conv: dropout_conv, p_keep_hidden: dropout_hidden})
				if cnt % display_iters == 0:
					summary = sess.run(merged, feed_dict={
						X: trX[start:end], Y: trY[start:end], p_keep_conv: 1.0, p_keep_hidden: 1.0
					})
					writer.add_summary(summary, cnt * batch_size)
				cnt += 1

			test_indices = np.arange(len(teX))
			np.random.shuffle(test_indices)
			test_indices = test_indices[0:test_size]
			test_acc = sess.run(accuracy, feed_dict={
				X: teX[test_indices], Y:teY[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0
			})
			print(i, test_acc)


def make_hparam_string(hparam):
	return "lr_%.0E_bs_%d_conv_p_%.0E, hidden_p_%.0E" % (hparam[0], hparam[1], hparam[2], hparam[3])


if __name__ == '__main__':
	learning_rate = 1E-3
	p_keep_conv = 0.8  # 1.0
	for batch_size in [32, 64, 128, 256]:
		for p_keep_hidden in [0.6, 0.7, 0.8]:
			hparam = make_hparam_string([learning_rate, batch_size, p_keep_conv, p_keep_hidden])
			print("Starting run for %s" % hparam)
			# run model with new hparam
			model(learning_rate, batch_size, p_keep_conv, p_keep_hidden, hparam)
	print("Done Train")
