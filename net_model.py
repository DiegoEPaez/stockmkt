import tensorflow as tf
import logging
import numpy as np


class RNNModel:

    def __init__(self, n_epochs, column_predict):
        self.tf_X = None
        self.tf_y = None
        self.tf_training = None
        self.tf_saver = None
        self.tf_training_op = None
        self.tf_init = None
        self.tf_outputs = None
        self.tf_loss = None
        self.n_epochs = n_epochs
        self.column_predict = column_predict

    def set_n_epochs(self, n_epochs):
        self.n_epochs = n_epochs

    def define_net(self, no_cols, n_steps, keep=1.0, lambda_l2_reg=0):
        n_inputs = no_cols
        n_outputs = 1
        n_neurons = 100
        n_layers = 3

        # In case this method was called before
        tf.reset_default_graph()

        self.tf_X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        self.tf_y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
        self.tf_training = tf.placeholder_with_default(False, shape=(), name='training')

        keep_prob_val = keep
        keep_prob = tf.cond(self.tf_training, lambda: tf.constant(keep_prob_val), lambda: tf.constant(1.0))

        he_init = tf.contrib.layers.variance_scaling_initializer()
        # gru_cells = [tf.contrib.rnn.GRUCell(num_units=n_neurons, kernel_initializer=he_init) for layer in range(n_layers)]
        gru_cells = [tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.GRUCell(num_units=n_neurons, kernel_initializer=he_init),
            input_keep_prob=keep_prob, output_keep_prob=keep_prob)
            for _ in range(n_layers)]

        multi_cell = tf.contrib.rnn.MultiRNNCell(gru_cells)
        rnn_outputs, states = tf.nn.dynamic_rnn(multi_cell, self.tf_X, dtype=tf.float32)

        learning_rate = 0.001

        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs, kernel_initializer=he_init)
        self.tf_outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

        self.tf_loss = tf.reduce_mean(tf.square(self.tf_outputs - self.tf_y))

        # Regularization
        l2 = lambda_l2_reg * sum(
            tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )
        self.tf_loss += tf.cond(self.tf_training, lambda: l2, lambda: tf.constant(0.0))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.tf_training_op = optimizer.minimize(self.tf_loss)

        self.tf_init = tf.global_variables_initializer()
        self.tf_saver = tf.train.Saver()

    def next_batch(self, epoch, batch_size, X_data, y_data):
        np.random.seed(epoch)
        indices = np.random.randint(X_data.shape[0], size=batch_size)
        X_batch = X_data[indices, :, :]
        y_batch = y_data[indices, :, :]
        return X_batch, y_batch

    def train_model(self, X_train, y_train, batch_size=50):

        with tf.Session() as sess:
            self.tf_init.run()

            for epoch in range(self.n_epochs):
                X_batch, y_batch = self.next_batch(epoch, batch_size, X_train, y_train)
                sess.run(self.tf_training_op, feed_dict={self.tf_X: X_batch, self.tf_y: y_batch, self.tf_training: True})
                if epoch % 100 == 0:
                    mse = self.tf_loss.eval(feed_dict={self.tf_X: X_batch, self.tf_y: y_batch})
                    logging.info(str(epoch) + "\tMSE:" + str(mse))

            self.tf_saver.save(sess, "outputs/model/rnn_model_" + self.column_predict + ".ckpt")

    def pred_rnn(self,  X):
        with tf.Session() as sess:
            # restore model
            self.tf_saver.restore(sess, "outputs/model/rnn_model_" + self.column_predict + ".ckpt")

            y_pred = sess.run(self.tf_outputs, feed_dict={self.tf_X: X, self.tf_training: False})

        return y_pred
