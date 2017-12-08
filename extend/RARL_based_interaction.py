"""
Extended Analysis: RARL on BDD pedestrian detection dataset
"""

import tensorflow as tf

from reimplementation.utilities import *


def build_mlp(input_placeholder,
            output_size,
            scope,
            n_layers=2,
            size=50,
            activation=tf.nn.relu,
            output_activation=tf.sigmoid
            ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out


def inject_noise(value, noise):
    if np.all(np.abs(noise) <= 0.2):
        return value + noise
    else:
        return value


class ProModel:
    def __init__(self,
                sess,
                name,
                n_layers=2,
                size=50,
                activation=tf.nn.relu,
                output_activation=tf.sigmoid,
                learning_rate=5e-3,
                ):
        self.sess = sess
        self.state = tf.placeholder(shape=[None, 4], name=name+"input", dtype=tf.float32)
        self.next = tf.placeholder(shape=[None, ], name=name+"next", dtype=tf.float32)
        self.pred_next = build_mlp(self.state, 1, name+"pro", n_layers, size, activation, output_activation)
        self.loss = tf.reduce_mean((self.pred_next - self.next)**2)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def train(self, x, y):
        ind = np.random.choice(len(x), len(x), replace=False)
        _, loss = self.sess.run([self.update_op, self.loss], feed_dict={self.state:x[ind], self.next:y[ind]})
        return loss

    def predict(self, x):
        return np.reshape(self.sess.run(self.pred_next, feed_dict={self.state:x}), -1)

    def evaluate(self, x, y):
        return self.sess.run([self.loss], feed_dict={self.state:x, self.next:y})


class AdvModel:
    def __init__(self,
                sess,
                name,
                n_layers=2,
                size=50,
                activation=tf.nn.relu,
                output_activation=tf.sigmoid,
                learning_rate=5e-3,
                ):
        self.sess = sess
        self.state = tf.placeholder(shape=[None, 4], name=name+"input", dtype=tf.float32)
        self.pred_next = tf.placeholder(shape=[None, ], name=name+"prediction", dtype=tf.float32)
        self.next = tf.placeholder(shape=[None, ], name=name+"next", dtype=tf.float32)
        self.pred_adv = build_mlp(self.state, 4, name+"adv", n_layers, size, activation, output_activation)
        self.clipped_adv = tf.clip_by_value(self.pred_adv, -0.2, 0.2)
        self.loss = -tf.reduce_mean((self.pred_next - self.next)**2*tf.reduce_sum(tf.abs(self.clipped_adv), axis=1)) + 0.01 * tf.nn.l2_loss(self.clipped_adv)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def train(self, x, y, pred_y):
        ind = np.random.choice(len(x), len(x), replace=False)
        _, loss = self.sess.run([self.update_op, self.loss], feed_dict={self.state:x[ind], self.next:y[ind], self.pred_next:pred_y[ind]})
        return loss

    def predict(self, x):
        return self.sess.run(self.clipped_adv, feed_dict={self.state:x})


def train_baseline_model(sess, baseline_m, iterations=25000):
    data = get_training_data(human=True)
    x = data[:,:-1]
    y = data[:, -1]
    mean = np.mean(y)
    std = np.std(y)
    norm_y = (y - mean)/std

    losses = []

    for i in range(iterations):
        loss = baseline_m.train(x, y)
        losses.append(loss)
        print("iter "+str(i)+" loss: " + str(loss))

    return baseline_m, np.array(losses)


def train_RARL(sess, pro, adv, iterations=25000):
    data = get_training_data(human=True)
    x = data[:,:-1]
    y = data[:, -1]
    mean = np.mean(y)
    std = np.std(y)
    norm_y = (y - mean)/std

    losses = []

    for i in range(iterations):
        noise = adv.predict(x)
        x_with_noise = inject_noise(x, noise)
        pro_loss = pro.train(x_with_noise, y)
        pred_y = pro.predict(x_with_noise)
        adv_loss = adv.train(x, y, pred_y)
        loss = pro.evaluate(x, y)
        losses.append(loss)
        print("iter "+str(i)+" pro loss: " + str(pro_loss) + " adv loss: " +str(adv_loss))

    return pro, adv, np.array(losses)


def main():
    sess = tf.Session()
    baseline_m = ProModel(sess, "baseline_pro_")
    pro_m = ProModel(sess, "pro_")
    adv_m = AdvModel(sess, "adv_")

    sess.__enter__()
    tf.global_variables_initializer().run()
    pro_m, adv_m, rarl_losses = train_RARL(sess, pro_m, adv_m, 1600)

    baseline_m, baseline_losses = train_baseline_model(sess, baseline_m, 1600)

    return baseline_m, pro_m, adv_m, baseline_losses, rarl_losses
