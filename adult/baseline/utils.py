import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy

class ClassifierGraph(keras.Model):

    def __init__(self, n_hiddens, num_classes, input_shape = (39,), seed = 1):
        super(ClassifierGraph, self).__init__()
        tf.random.set_seed(seed)
        self.Layers = []
        self.Layers.append(keras.layers.Dense(n_hiddens[0], activation = tf.nn.relu, name = 'layer-1', input_shape = input_shape))
        if len(n_hiddens) > 1:
            for i, n in enumerate(n_hiddens[1:], 2):
                self.Layers.append(keras.layers.Dense(n, activation = tf.nn.relu,\
                     name = f'layer-{i}'))
        #self.layer1 = keras.layers.Dense(n_hidden1, activation = tf.nn.relu, name = 'layer-1', input_shape = input_shape)
        self.Layers.append(keras.layers.Dense(num_classes, activation = tf.nn.softmax, name = 'output'))
        self.model = keras.models.Sequential(self.Layers)

    def call(self, x, predict = False):
        x = self.model(x)
        #x, _ = tf.linalg.normalize(x, ord = 1, axis = 1)
        return tf.cast(tf.argmax(x, axis = 1), dtype = tf.float32) if predict else x


def EntropyLoss(y, prob):
    return -2*tf.reduce_mean(tf.math.multiply(y, tf.math.log(prob)))




def _accuracy(y, ypred):
    acc = tf.cast(tf.equal(y, ypred), dtype = tf.float32)
    return tf.reduce_mean(acc)




def unprotected_direction(x, sensetive_directions):
    x = x - x @ tf.linalg.matrix_transpose(sensetive_directions) @ sensetive_directions
    return x




