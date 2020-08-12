import numpy as np
import tensorflow as tf
import compas_data as compas
from sklearn import linear_model
import scipy

def standardize(x):
    return (x - np.mean(x))/np.std(x)

def get_data(random_state = 0):
    # Extracting compas data
    x_train, x_test, y_train, y_test, y_sex_train, y_sex_test,\
        y_race_train, y_race_test, _ = compas.get_compas_train_test(random_state = random_state)
    #x_train, x_test = x_train[:, 2:], x_test[:, 2:]

    
    # casting to tensor
    x_train, x_test = tf.cast(x_train, dtype=tf.float32), tf.cast(x_test, dtype = tf.float32)
    y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
    y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)

    return x_train, x_test, y_train, y_test, y_sex_train, y_race_train, y_sex_test, y_race_test