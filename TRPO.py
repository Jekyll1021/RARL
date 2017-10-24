import tensorflow as tf
import numpy as np
import gym

def build_mlp(input_placeholder, 
				output_size,
				scope, 
				n_layers=2, 
				size=500, 
				activation=tf.tanh,
				output_activation=None
				):
	out = input_placeholder
	with tf.variable_scope(scope):
		for _ in range(n_layers):
			out = tf.layers.dense(out, size, activation=activation)
			out = tf.layers.dense(out, output_size, activation=output_activation)
	return out

class TRPOModel:
	def __init__(self, learning_rate, n_layers=2, size, activation=tf.tanh, output_activation=None, env, sess, pro=True):

		self.learning_rate = learning_rate
		self.sess = sess

		self.states = tf.placeholder(shape=[None, self.env.observation_space.shape[0]], name="states", dtype=tf.float32)
		if pro:
			act_dim = env.sample_action().pro.shape[0]
		else:
			act_dim = env.sample_action().adv.shape[0]
		self.actions = tf.placeholder(shape=[None, act_dim,] name="actions", dtype=tf.float32)
		self.advantages = tf.placeholder(shape=[None], name="advantages", dtype=tf.float32)

		# TODO: substitute this part with TRPO updating policies. This is Natural Gradient Descent for now. 
		self.pred_actions = build_mlp(self.states, act_dim, "pred_action", n_layers, size, activation, output_activation)
        logstd = tf.get_variable("logstdev", [ac_dim], initializer=tf.zeros_initializer()) # logstd should just be a trainable variable, not a network output.
        self.sampled_actions = self.pred_actions + tf.random_normal(shape=tf.shape(self.pred_actions))*tf.exp(logstd)
        diff_mean = self.actions - self.pred_actions
        logprob_n = -1/2*tf.reduce_sum(diff_mean**2/(tf.exp(logstd)**2), axis=1) - ac_dim*0.5*tf.log(tf.constant(2*np.pi)) - tf.reduce_sum(logstd) # Hint: Use the log probability under a multivariate gaussian. 

        self.loss = -tf.reduce_sum(advantages * logprob_n) # Loss function that we'll differentiate to get the policy gradient.
    	self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    	self.saver = tf.train.Saver()

    def sample(self):
    	# TODO: TRPO specific
    	pass

    def learn(self, states, actions, advantages):
    	_, loss = self.sess.run([self.update_op, self.loss], feed_dict={self.states:states, self.actions:actions, self.advantages:advantages})
    	return loss

    def predict(self, state):
    	return self.sess.run(self.sampled_actions, feed_dict={self.states:state})

    def save(self, path):
    	self.saver.save(self.sess, path)

    def load(self, path):
    	self.saver.restore(self.sess, path)