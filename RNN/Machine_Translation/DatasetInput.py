import tensorflow as tf
import numpy as np
import os


MAX_LEN = 50  # 限制句子的最大单词数量
SOS_ID = 1  # 目标语言词汇表中<sos>的ID


# 使用Dataset  API从文件中读取一个语言的数据
# 数据格式为每行一句话，dataset中一个元素为一句话，单词已转为单词编号
def make_dataset(file_path):
	dataset = tf.data.TextLineDataset(file_path)
	dataset = dataset.map(lambda string: tf.string_split([string]).values)
	dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
	# 统计每个元素（句子单词编号列表）的单词数量，并与句子内容一起放入dataset中
	dataset = dataset.map(lambda x: (x, tf.size(x)))
	return dataset


# 从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和batching操作
def make_src_trg_dataset(src_path, trg_path, batch_size):
	src_data = make_dataset(src_path)
	trg_data = make_dataset(trg_path)
	dataset = tf.data.Dataset.zip((src_data, trg_data))

	# 删除内容为空（只包含<eos>）的句子和长度过长的句子
	def filter_len(src_tuple, trg_tuple):
		((src_list, src_len), (trg_list, trg_len)) = (src_tuple, trg_tuple)
		src_ok = tf.logical_and(
			tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN)
		)
		trg_ok = tf.logical_and(
			tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN)
		)
		return tf.logical_and(src_ok, trg_ok)
	dataset = dataset.filter(filter_len)

	# seq2seq模型中，解码器输入输出训练数据是前后延迟一个单位的句子
	# trg_input: "<sos> X Y Z"
	# trg_label: "X Y Z <EOS>"
	# 之前句子形式是"X Y Z <EOS>", 现在要生成"<SOS> X Y Z"形式
	def make_trg_input(src_tuple, trg_tuple):
		((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
		trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
		return ((src_input, src_len), (trg_input, trg_label, trg_len))
	dataset = dataset.map(make_trg_input)
	dataset.shuffle(10000)
	# 规定填充后输出的数据维度
	padding_shapes = (
		(tf.TensorShape([None]),  # 源句子是长度未知的向量
			tf.TensorShape([])),  # 源句子长度是单个数字
		(tf.TensorShape([None]),  # 目标句子（解码器输入）是长度未知的向量
			tf.TensorShape([None]),  # 目标句子（解码器输出）是长度未知的向量
			tf.TensorShape([]))  # 目标句子长度是单个数字
	)
	batched_dataset = dataset.padded_batch(batch_size, padding_shapes)
	return batched_dataset
