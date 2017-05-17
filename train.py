
import cPickle as pickle
import gzip
import pdb

import tensorflow as tf
import numpy as np

from models.model import Model
from utils.config import TrainConfig
from utils.data_helper import batch_iter

# load MNIST dataset
fh = gzip.open('data/mnist.pkl.gz', 'rb')
training_data, validation_data, _ = pickle.load(fh)
fh.close()

# setup training configurations
tr_config = TrainConfig('model1.cfg')
tr_config.show_config()
 
# setup model input tensors
x = tf.placeholder(tf.float32, [None, 784])
y_hat = tf.placeholder(tf.int32, [None,])
keep_prob = tf.placeholder(tf.float32)
input_tensors = {}
input_tensors['x'] = x
input_tensors['y_hat'] = y_hat
input_tensors['keep_prob'] = keep_prob

# init model
model = Model(tr_config, input_tensors)
sess = tf.Session()
model.init_vars(sess)

# python3
#batches = batch_iter(list(zip(training_data[0], training_data[1])), 
batches = batch_iter(zip(training_data[0], training_data[1]), 
                     tr_config.batch_size, tr_config.num_epochs)
step = 0
for batch in batches:
    x_batch, y_hat_batch = zip(*batch)
    x_batch, y_hat_batch = np.array(x_batch), np.array(y_hat_batch)

    model.train(sess, x_batch, y_hat_batch, tr_config.keep_prob)
    step += 1
    if step % tr_config.eval_every == 0:
        val_batches = batch_iter(zip(validation_data[0], validation_data[1]),
                                 tr_config.batch_size, 1)
        n_err = 0
        for batch in val_batches:
            x_batch, y_hat_batch = zip(*batch)
            x_batch, y_hat_batch = np.array(x_batch), np.array(y_hat_batch)
            
            y_batch = model.get_tensor_val(sess, ['y'], x_batch, y_hat_batch)
            #pdb.set_trace()
            y_batch = np.argmax(y_batch[0], axis=1)
            n_err += np.sum( y_batch != y_hat_batch)
        error_rate = float(n_err) / len(validation_data[0])
        print("step: %i, error_rate: %f" % (step, error_rate))

