import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import classifier as cl
import utils
import time
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import scipy
plt.ioff()
from tensorflow import keras


seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)


seeds = np.load('../seeds.npy')


for i in range(10):
    data_seed = seeds[i, 0]
    expt_seed = seeds[i, 1]
    dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = data_seed)

    x_unprotected_train, x_protected_train = dataset_orig_train.features[:, :39], dataset_orig_train.features[:, 39:]
    x_unprotected_test, x_protected_test = dataset_orig_test.features[:, :39], dataset_orig_test.features[:, 39:]
    y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))

    # Casing to tensor 
    y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
    x_unprotected_train, x_unprotected_test = tf.cast(x_unprotected_train, dtype = tf.float32), tf.cast(x_unprotected_test, dtype = tf.float32)
    y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)



    print(f'Running data seed {data_seed} and expt seed {expt_seed}')
    init_graph = utils.ClassifierGraph([50,], 2, input_shape=(39, ), seed_data=data_seed, seed_model=expt_seed)
    graph = cl.Classifier(init_graph, x_unprotected_train, y_train, num_steps = 8000, seed = expt_seed) # use for unfair algo
    #graph.model._set_inputs((-1, 39))
    inputs = keras.Input((39,))

    outputs = graph(inputs)
    model = keras.Model(inputs, outputs)
    model.save(f'graphs/graph_{data_seed}_{expt_seed}')
