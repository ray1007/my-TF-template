import os

import numpy as np
import tensorflow as tf

def fc(x, n_inputs, n_outputs, scope, use_bias=True, act=tf.nn.relu):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", [n_inputs, n_outputs],
          initializer=tf.truncated_normal_initializer(stddev=0.1))
        if use_bias:
            b = tf.get_variable("b", [n_outputs],
              initializer=tf.truncated_normal_initializer(stddev=0.1))
            return act(tf.matmul(x, w) + b)
        else:
            return act(tf.matmul(x, w))

def conv(x, kernel_shape, scope_name, strides=[1,1,1,1], padding='SAME', act=tf.nn.relu6):
    with tf.variable_scope(scope_name) as scope:
        kernel = tf.get_variable("kernel", kernel_shape,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable("biases",[kernel_shape[-1]],initializer=tf.constant_initializer(0.5))
        conv = tf.nn.conv2d(x, kernel, strides, padding=padding)
        bias = tf.nn.bias_add(conv, biases)
        x = act(bias, name=scope.name)
    return x

class Model:
    def __init__(self, config, input_tensors):
        print "model.py :: __init__()"
        x = input_tensors['x']
        y_hat = input_tensors['y_hat']
        keep_prob = input_tensors['keep_prob']

        self.x = x
        self.y_hat = y_hat
        self.keep_prob = keep_prob

        y_hat = tf.one_hot(y_hat, 10)
        #images = tf.placeholder(tf.float32, (None, 60, 60, 1))
        #keep_prob = tf.placeholder(tf.float32)

        x_drop = tf.nn.dropout(x, keep_prob)
        with tf.variable_scope('model'):
            x = fc(x, 784, 100, "fc1")
            y = fc(x, 100, 10, "fc2", act=tf.nn.sigmoid)
    
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_sum((y - y_hat)** 2) / config.batch_size
            #xent = tf.nn.sigmoid_cross_entropy_with_logits(y, y_hat)
            #self.loss = tf.reduce_sum(xent) / config.batch_size
        '''
        with tf.variable_scope('learning_rate'):
            lr_step = tf.Variable(0, trainable=False)
            decay_lr = tf.assign_add(lr_step, 1)
            self.lr = tf.train.exponential_decay(0.001, lr_step, 1, 0.85)
            self.lr = 0.001
        '''
        with tf.variable_scope('train'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
            #optimizer = tf.train.AdamOptimizer(config.learning_rate)
            #optimizer = tf.train.AdagradOptimizer(0.0001)
            self.train_op = optimizer.minimize(self.loss)
      
        with tf.variable_scope('gradient_computing'):
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = gvs
            #capped_gvs = [(tf.clip_by_norm(grad, 5), var)\
            #             for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs)
      
        self.tensor_list= {}
        self.tensor_list['loss'] = self.loss
        self.tensor_list['y'] = y

        self.saver = tf.train.Saver(max_to_keep=100)

    def init_vars(self, sess):
        sess.run(tf.global_variables_initializer())

    def save_model(self, sess, path, step):
        self.saver.save(sess, path, global_step=step)

    def load_model(self, sess, model_path):
        self.saver.restore(sess, model_path)

    def train(self, sess, x, y_hat, keep_prob):
        sess.run(self.train_op, {
            self.x: x,
            self.y_hat: y_hat,
            self.keep_prob: keep_prob})

    def get_tensor_val(self, sess, tensor_name_list, x, y_hat):
        fetch_tensors = [ self.tensor_list[name] for name in tensor_name_list ]
        return sess.run(fetch_tensors, {
            self.x: x,
            self.y_hat: y_hat})


