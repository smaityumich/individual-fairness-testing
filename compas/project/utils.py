import tensorflow as tf
from tensorflow import keras
import numpy as np





class Project(keras.layers.Layer):
    
    def __init__(self, w):
        super(Project, self).__init__()
        self.w = tf.Variable(shape = (2, 5),initial_value=w,\
                               trainable=False)
        self.input_spec = tf.keras.layers.InputSpec(shape=(None, 5))

    def call(self, x):
        return unprotected_direction(x, self.w)





class ClassifierGraph(keras.Model):

    def __init__(self, n_hiddens, num_classes, sensetive_directions, input_shape = (5,), seed_model = 1):
        super(ClassifierGraph, self).__init__()
        tf.random.set_seed(seed_model)
        self.Layers = [Project(sensetive_directions),]
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




