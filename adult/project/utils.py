import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy


import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import utils
import time
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import scipy
plt.ioff()
import sys


class Project(keras.layers.Layer):
    
    def __init__(self, w):
        super(Project, self).__init__()
        self.w = tf.Variable(shape = (2, 39),initial_value=w,\
                               trainable=False)
        self.input_spec = tf.keras.layers.InputSpec(shape=(None, 39))

    def call(self, x):
        return unprotected_direction(x, self.w)





class ClassifierGraph(keras.Model):

    def __init__(self, n_hiddens, num_classes, input_shape = (39,), seed_data = 1, seed_model = 1):
        super(ClassifierGraph, self).__init__()
        tf.random.set_seed(seed_model)
        dataset_orig_train, _ = preprocess_adult_data(seed = seed_data)

        x_unprotected_train, x_protected_train = dataset_orig_train.features[:, :39], dataset_orig_train.features[:, 39:]
        


    



        sensetive_directions = []
        protected_regression = linear_model.LogisticRegression(fit_intercept = True)
        protected_regression.fit(x_unprotected_train, x_protected_train[:, 0])
        sensetive_directions.append(protected_regression.coef_.reshape((-1,)))
        protected_regression.fit(x_unprotected_train, x_protected_train[:, 1])
        sensetive_directions.append(protected_regression.coef_.reshape((-1,)))
        sensetive_directions = np.array(sensetive_directions)

        sensetive_directions = scipy.linalg.orth(sensetive_directions.T).T
        for i, s in enumerate(sensetive_directions):
            while np.linalg.norm(s) != 1:
                s = s/ np.linalg.norm(s)
            sensetive_directions[i] = s
        sensetive_directions = tf.cast(sensetive_directions, dtype = tf.float32)

        
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




def _accuracy(y, prob):
    ypred = tf.cast(tf.argmax(prob, axis = 1), dtype = tf.float32)
    acc = tf.cast(ypred, dtype = tf.float32)-tf.cast(y[:, 1], dtype = tf.float32)
    return tf.reduce_mean(tf.abs(acc))





def unprotected_direction(x, sensetive_directions):
    x = x - x @ tf.linalg.matrix_transpose(sensetive_directions) @ sensetive_directions
    return x




