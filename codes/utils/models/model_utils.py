import tensorflow as tf
import os
import numpy as np
import random
from tensorflow.keras import backend as K


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def sampling(args):
    z_mean, z_log_sigma = args
    # batch_size = tf.shape(z_mean)[0]
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=1.0)
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon


def vae_loss(original, out, z_log_sigma, z_mean):
    reconstruction = K.mean(K.square(original - out)) * original.shape[1]
    kl_loss = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
    return reconstruction + kl_loss
