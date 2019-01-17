import tensorflow as tf
import numpy as np
import os
import DatasetInput as DI
import codecs


# Global Param
SRC_TRAIN_DATA = 'data/en.train'
TRG_TRAIN_DATA = 'data/zh.train'
CHECKPOINT_PATH = 'model_unionG/seq2seq_attention_ckpt'  # checkpoint保持路径
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
BATCH_SIZE = 100
NUM_EPOCH = 5
KEEP_PROB = 0.8  # 训练时dropout层节点保留的概率
MAX_GRAD_NORM = 5  # 梯度截断上限值
SHARE_EMB_AND_SOFTMAX = True  # embedding层与softmax层是否参数共享
SOS_ID = 1
EOS_ID = 2
LOG_DIR = 'LOG/'


# Seq2seq model
class S2SModel(object):
	def __init__(self):
		# 定义输入输出
		with tf.variable_scope("INPUT"):
			self.src_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='src_input')
			self.src_len = tf.placeholder(dtype=tf.int32, shape=[None], name='src_len')
			self.trg_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='trg_input')
			self.trg_len = tf.placeholder(dtype=tf.int32, shape=[None], name='trg_len')
		with tf.variable_scope("LABEL"):
			self.trg_label = tf.placeholder(dtype=tf.int32, shape=[None, None], name='trg_label')
		# 定义编码器和解码器所使用的LSTM结构, 无dropout
		with tf.variable_scope("ENCODE_PARAM"):
			self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
			self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
		with tf.variable_scope("DECODE_PARAM"):
			self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([
				tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)
			])
			self.attention_cell = None

		# 为源语言和目标语言分别定义词向量
		with tf.variable_scope("ENCODE_PARAM"):
			self.src_embedding = tf.get_variable("src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
		with tf.variable_scope("DECODE_PARAM"):
			self.trg_embedding = tf.get_variable("trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

		# 定义softmax层
		with tf.variable_scope("DECODE_PARAM/SOFTMAX"):
			if SHARE_EMB_AND_SOFTMAX:
				self.softmax_weight = tf.transpose(self.trg_embedding, name="weight")
			else:
				self.softmax_weight = tf.get_variable("weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
			self.softmax_bias = tf.get_variable("bias", [TRG_VOCAB_SIZE])

	# 在forward函数中定义模型的前向计算图
	# (([src_input], src_len), ([trg_input], [trg_label], trg_len))是dataset中的一个元素
	def forward(self):

		batch_size = tf.shape(self.src_input)[0]

		# 将输入单词映射为词向量
		with tf.variable_scope("ENCODE/EMBEDDING"):
			src_emb = tf.nn.embedding_lookup(self.src_embedding, self.src_input, name='embedding_out')
		with tf.variable_scope("DECODE/EMBEDDING"):
			trg_emb = tf.nn.embedding_lookup(self.trg_embedding, self.trg_input, name='embedding_out')

		with tf.variable_scope("ENCODE/DROPOUT"):
			src_emb = tf.nn.dropout(src_emb, keep_prob=KEEP_PROB, name='dropout_out')
		with tf.variable_scope("DECODE/DROPOUT"):
			trg_emb = tf.nn.dropout(trg_emb, keep_prob=KEEP_PROB, name='dropout_out')

		# Encoder用不到cell output张量
		with tf.variable_scope("ENCODE/LSTM_CELL"):
			enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=self.enc_cell_fw,
				cell_bw=self.enc_cell_bw,
				inputs=src_emb,
				sequence_length=self.src_len,
				initial_state_fw=self.enc_cell_fw.zero_state(batch_size, tf.float32),
				initial_state_bw=self.enc_cell_bw.zero_state(batch_size, tf.float32),
				dtype=tf.float32
			)
			# shape: [batch_size, max_time, HIDDEN_SIZE*2]
			enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], axis=-1)
		# Attention机制
		with tf.variable_scope("ATTENTION"):
			# 选择注意力权重的计算模型
			# BahdanauAttention是使用一个隐层的前馈神经网络
			# memory_sequence_length是一个维度为[batch_size]的张量,代表batch
			# 中每个句子的长度,Attention需要根据这个信息把这个填充位置的注意力权重设置为0
			# 这是个注意力机制的类,现给他实例化
			attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
				num_units=HIDDEN_SIZE,
				memory=enc_outputs,
				memory_sequence_length=self.src_len
			)
		# Decoder需要Encoder最后状态作为初始状态, 其本身不需要最后状态
		with tf.variable_scope("DECODE/ATT_CELL"):
			self.attention_cell = tf.contrib.seq2seq.AttentionWrapper(
				cell=self.dec_cell,
				attention_mechanism=attention_mechanism,
				attention_layer_size=HIDDEN_SIZE
			)
		with tf.variable_scope("DECODE/LSTM_CELL"):
			dec_output, _ = tf.nn.dynamic_rnn(
				cell=self.attention_cell,
				inputs=trg_emb,
				sequence_length=self.trg_len,
				initial_state=self.attention_cell.zero_state(batch_size, tf.float32),
				dtype=tf.float32
			)

		# 计算解码器每一步的log perplexity. 这一步并没有将填充位置给屏蔽掉
		with tf.variable_scope("DECODE/OUTPUT"):
			# output: [batch_size*time_step, hidden_size]
			output = tf.reshape(dec_output, shape=[-1, HIDDEN_SIZE])
			logits = tf.nn.bias_add(tf.matmul(output, self.softmax_weight), self.softmax_bias)
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
				labels=tf.reshape(self.trg_label, [-1]),
				logits=logits,
				name='log_perplexity'
			)

		# 计算每个batch的总损失,句子平均损失,单词平均损失
		# 将填充位置的权重设置为0屏蔽掉, 以避免无效位置的预干扰模型训练
		with tf.variable_scope("LOSS"):
			label_weights = tf.sequence_mask(self.trg_len, maxlen=tf.shape(self.trg_label)[1], dtype=tf.float32)
			label_weights = tf.reshape(label_weights, [-1])
			# 一个batch中所有句子的总损失
			cost = tf.reduce_sum(tf.multiply(loss, label_weights), name='batch_cost')
			# 一个batch中每个单词的平均损失
			with tf.variable_scope("avg_token_cost"):
				cost_per_token = cost / tf.to_float(tf.reduce_sum(self.trg_len))
				tf.summary.scalar("cost_per_token", cost_per_token)
			# 一个batch中每个句子的平均损失, 训练目标
			with tf.variable_scope("avg_seq_cost"):
				cost_per_seq = cost / tf.to_float(batch_size)
				tf.summary.scalar("cost_per_seq", cost_per_seq)

		# 梯度下降, 训练操作
		with tf.variable_scope("Train"):
			trainable_var = tf.trainable_variables()
			grads = tf.gradients(cost_per_seq, trainable_var)
			grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
			train_op = optimizer.apply_gradients(zip(grads, trainable_var))
		return cost_per_token, train_op

	def inference(self):
		batch_size = tf.shape(self.src_len)[0]
		# 设置解码的最大步数
		MAX_DEC_LEN = 100
		enc_state = self.attention_cell.zero_state(batch_size, tf.float32)
		with tf.variable_scope("DECODE/LSTM_CELL"):
			init_tensor = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
			init_tensor = init_tensor.write(0, SOS_ID)
			init_loop_var = (enc_state, init_tensor, 0)

			def continue_cond(state, trg_tensor, step):
				return tf.reduce_all(tf.logical_and(
					tf.not_equal(trg_tensor.read(step), EOS_ID),
					tf.less(step, MAX_DEC_LEN-1)
				))

			def loop_body(state, trg_tensor, step):
				trg_input = [trg_tensor.read(step)]
				trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input, name='embedding_out')
				dec_output, next_state = self.attention_cell.call(inputs=trg_emb, state=state)
				output = tf.reshape(dec_output, [-1, HIDDEN_SIZE])
				logit = tf.nn.bias_add(tf.matmul(output, self.softmax_weight), self.softmax_bias)
				next_word_cnt = tf.argmax(logit, axis=1, output_type=tf.int32)
				trg_tensor = trg_tensor.write(step+1, next_word_cnt[0])
				return next_state, trg_tensor, step+1

			state, trg_tensor, step = tf.while_loop(
				cond=continue_cond,
				body=loop_body,
				loop_vars=init_loop_var,
				name='WHILE_LOOP'
			)
			return trg_tensor.stack()


def run_epoch(sess, cost_op, train_op, saver, step, merged, writer, model, iter):
	while True:
		try:
			(src_input, src_len), (trg_input, trg_label, trg_len) = sess.run(iter.get_next())
			cost, _ = sess.run([cost_op, train_op], feed_dict={
					model.src_input: src_input,
					model.src_len: src_len,
					model.trg_input: trg_input,
					model.trg_label: trg_label,
					model.trg_len: trg_len
			})
			if step % 10 == 0:
				print("After %d steps, per token cost is %.3f" % (step, cost))
				summary = sess.run(merged, feed_dict={
					model.src_input: src_input,
					model.src_len: src_len,
					model.trg_input: trg_input,
					model.trg_label: trg_label,
					model.trg_len: trg_len
				})
				writer.add_summary(summary, step*BATCH_SIZE)
			if step % 200 == 0:
				saver.save(sess, CHECKPOINT_PATH, global_step=step)
			step += 1
		except tf.errors.OutOfRangeError:
			break
	return step


if __name__ == '__main__':

	# 默认图
	tf.reset_default_graph()
	# 英文词汇表
	with codecs.open('data/en.vocab', 'r', 'utf-8') as f:
		hash_en = {line.strip(): i for i, line in enumerate(f)}
	# 中文词汇表
	with codecs.open('data/zh.vocab', 'r', 'utf-8') as f:
		hash_zh = [line.strip() for line in f]
	# 推断
	test_sequence = "we are family .".split()
	test_input = [hash_en[s] if s in hash_en else hash_en['<unk>'] for s in test_sequence]
	# 定义seq2seq模型
	with tf.variable_scope("Seq2Seq", reuse=None, initializer=tf.random_uniform_initializer(-0.05, 0.05)):
		train_model = S2SModel()
		# 训练数据集
		dataset = DI.make_src_trg_dataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
		iter = dataset.make_initializable_iterator()
		# (src_input, src_len), (trg_input, trg_label, trg_len) = iter.get_next()
		with tf.variable_scope("TR", reuse=None):
			cost_op, train_op = train_model.forward()
		with tf.variable_scope("INFER", reuse=None):
			output_op = train_model.inference()

	with tf.Session() as sess:
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(LOG_DIR+'att', sess.graph)
		saver = tf.train.Saver()
		step = 0
		sess.run(tf.global_variables_initializer())
		for i in range(NUM_EPOCH):
			print("EPOCH: %d" % (i+1))
			sess.run(iter.initializer)
			step = run_epoch(sess, cost_op, train_op, saver, step, merged, writer, train_model, iter)
		# saver = tf.train.Saver()
		# saver.restore(sess, 'model/seq2seq_ckpt-8000')
		output = sess.run(output_op, feed_dict={
			train_model.src_input: [test_input],
			train_model.src_len: [len(test_input)]
		})
		pred_seq = [hash_zh[i] for i in output]
		print(pred_seq)

