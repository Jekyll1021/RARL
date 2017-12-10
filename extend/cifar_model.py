import tensorflow as tf 
import numpy as np
from cifar_input import *

def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.
	Args:
	name: name of the variable
	shape: list of ints
	initializer: initializer for Variable
	Returns:
	Variable Tensor
	"""
	with tf.device('/cpu:0'):
		dtype = tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd):
	"""Helper to create an initialized Variable with weight decay.
	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.
	Args:
	name: name of the variable
	shape: list of ints
	stddev: standard deviation of a truncated Gaussian
	wd: add L2Loss weight decay multiplied by this float. If None, weight
	    decay is not added for this Variable.
	Returns:
	Variable Tensor
	"""
	dtype = tf.float32
	var = _variable_on_cpu(
		name,
		shape,
		tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def build_cnn(input_placeholder, 
			output_size,
			name
			):
	with tf.variable_scope(name+'conv1') as scope:
		kernel = _variable_with_weight_decay('weights',
											shape=[5, 5, 3, 64],
											stddev=5e-2,
											wd=0.0)
		conv = tf.nn.conv2d(input_placeholder, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)

	# pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
					padding='SAME', name=name+'pool1')
	# norm1
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					name=name+'norm1')

	# conv2
	with tf.variable_scope(name+'conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
											shape=[5, 5, 64, 64],
											stddev=5e-2,
											wd=0.0)
		conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)

	# norm2
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					name=name+'norm2')
	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
					strides=[1, 2, 2, 1], padding='SAME', name=name+'pool2')

	# local3
	with tf.variable_scope(name+'local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		reshape = tf.reshape(pool2, [-1, 4096])
		weights = _variable_with_weight_decay('weights', shape=[4096, 384],
												stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

	# local4
	with tf.variable_scope(name+'local4') as scope:
		weights = _variable_with_weight_decay('weights', shape=[384, 192],
												stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

	# linear layer(WX + b),
	# We don't apply softmax here because
	# tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
	# and performs the softmax internally for efficiency.
	with tf.variable_scope(name+'softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights', [192, output_size],
										stddev=1/192.0, wd=0.0)
		biases = _variable_on_cpu('biases', [output_size],
									tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

	return softmax_linear

class ProModel:
	def __init__(self, 
				sess,
				name,
				learning_rate=5e-3,
				):
		self.sess = sess
		self.image = tf.placeholder(shape=[None, 32, 32, 3], name=name+"image", dtype=tf.float32)
		self.label = tf.placeholder(shape=[None, ], name=name+"label", dtype=tf.int64)
		self.logits = build_cnn(self.image, 10, name+"pro")
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
						labels=tf.one_hot(self.label, 10), logits=self.logits))
		self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), self.label), tf.float32))


	def train(self, x, y):
		_, loss, logits = self.sess.run([self.update_op, self.loss, self.logits], feed_dict={self.image:x, self.label:y})
		return loss, logits

	def predict(self, x):
		return self.sess.run(self.logits, feed_dict={self.image:x})

	def evaluate(self, x, y):
		return self.sess.run([self.accuracy], feed_dict={self.image:x, self.label:y})

class AdvModel:
	def __init__(self, 
				sess,
				name,
				learning_rate=5e-3,
				):
		self.sess = sess
		self.image = tf.placeholder(shape=[None, 32, 32, 3], name=name+"image", dtype=tf.float32)
		self.label = tf.placeholder(shape=[None, ], name=name+"label", dtype=tf.int64)
		self.pred_logits = tf.placeholder(shape=[None, 10], name=name+"label", dtype=tf.float32)
		self.noise = build_cnn(self.image, 32*32*3, name+"adv")
		self.clipped_adv = tf.clip_by_value(self.noise, -0.002, 0.002)
		self.loss = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(
						labels=tf.one_hot(self.label, 10), logits=self.pred_logits)*tf.reduce_sum(tf.abs(self.clipped_adv), axis=1)) + 0.01 * tf.nn.l2_loss(self.clipped_adv)
		self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

	def train(self, x, y, pred_y):
		_, loss = self.sess.run([self.update_op, self.loss], feed_dict={self.image:x, self.label:y, self.pred_logits:pred_y})
		return loss

	def predict(self, x):
		return self.sess.run(self.clipped_adv, feed_dict={self.image:x})

def train_baseline_model(sess, train_data, train_label, baseline_m, iterations=80001):

	losses = []

	accs = []

	for i in range(iterations):
		ind = np.random.choice(len(train_label), 128, replace=False)
		# x = random_crop_and_flip(transform(train_data[ind]), 2)
		x = whitening_image(transform(train_data[ind]))
		y = train_label[ind]
		loss, logits = baseline_m.train(x, y)
		losses.append(loss)
		print("iter "+str(i)+" loss: " + str(loss))

		if i % 100 == 0:
			acc = baseline_m.evaluate(x, y)
			print("iter "+str(i)+" acc: " + str(acc))
			accs.append(acc)

	return baseline_m, np.array(losses), np.array(accs)

def train_RARL(sess, train_data, train_label, pro, adv, iterations=80001):

	losses = []
	accs = []

	for i in range(iterations):
		ind = np.random.choice(len(train_label), 128, replace=False)
		# x = random_crop_and_flip(transform(train_data[ind]), 2)
		x = whitening_image(transform(train_data[ind]))
		y = train_label[ind]
		noise = adv.predict(x)
		x_with_noise = x + transform(noise)
		pro_loss, pred_y = pro.train(x_with_noise, y)
		adv_loss = adv.train(x, y, pred_y)
		loss = pro.evaluate(x, y)
		losses.append(loss)
		print("iter "+str(i)+" pro loss: " + str(pro_loss) + " adv loss: " +str(adv_loss))

		if i % 100 == 0:
			acc = pro.evaluate(x, y)
			print("iter "+str(i)+" acc: " + str(acc))
			accs.append(acc)


	return pro, adv, np.array(losses), np.array(accs)

def main():
	path_list = []
	for i in range(1, NUM_TRAIN_BATCH+1):
		path_list.append(full_data_dir + str(i))
	data, label = read_in_all_images(path_list)
	# data = prepare_train_data(data, 2)

	sess = tf.Session()
	baseline_m = ProModel(sess, "baseline_pro_")
	pro_m = ProModel(sess, "pro_")
	adv_m = AdvModel(sess, "adv_")

	sess.__enter__()
	tf.global_variables_initializer().run()
	pro_m, adv_m, rarl_losses, rarl_accs = train_RARL(sess, data, label, pro_m, adv_m)

	baseline_m, baseline_losses, baseline_accs = train_baseline_model(sess, data, label, baseline_m)

	test_x, test_y = read_validation_data()
	test_x = transform(test_x)

	b_acc = baseline_m.evaluate(test_x, test_y)
	pro_acc = pro_m.evaluate(test_x, test_y)

	noisy_x = test_x + transform(adv_m.predict(test_x))

	b_n_acc = baseline_m.evaluate(noisy_x, test_y)
	p_n_acc = pro_m.evaluate(noisy_x, test_y)

	np.save("baseline_loss", baseline_losses)
	np.save("rarl_loss", rarl_losses)
	np.save("baseline_acc", baseline_accs)
	np.save("rarl_acc", rarl_accs)

	return baseline_m, pro_m, adv_m, b_acc, pro_acc, b_n_acc, p_n_acc
