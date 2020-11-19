import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy



def EntropyLoss(y, prob):
    return -2*tf.reduce_mean(tf.math.multiply(y, tf.math.log(prob)))




def _accuracy(y, prob):
    ypred = tf.cast(tf.argmax(prob, axis = 1), dtype = tf.float32)
    acc = tf.cast(ypred-y[:, 1], dtype = tf.float32)
    return tf.reduce_mean(acc)



def unprotected_direction(x, sensetive_directions):
    x = x - x @ tf.linalg.matrix_transpose(sensetive_directions) @ sensetive_directions
    return x




