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
import json
from tensorflow import keras


def SimpleDense(variable):
    w, b = variable
    w = tf.cast(w, dtype = tf.float32)
    b = tf.cast(b, dtype = tf.float32)
    return lambda x: tf.matmul(x, w) + b



                

def sample_perturbation(data_point, regularizer = 100, learning_rate = 5e-2, num_steps = 200):
    """
    Calculates ratio between perturbed loss and original loss

    parameters: 
        data_point: tuple of x, y
            x: tensor of shape (d, )
            y: one-hot encoded tensor of shape (2, )
        regularizer (float): regularizer constant for fair metric
        learning_rate (float): step size for gradient ascend
        num_steps (int): number of steps in gradient ascend

    return:
        float; ratio of entropy losses for perturbed and original sample
    """
    x, y = data_point
    x = tf.reshape(x, (1, -1))
    y = tf.reshape(y, (1, -1))
    x_start = x
    for i in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x)
            prob = graph(x)
            perturb = utils.unprotected_direction(x-x_start, sensetive_directions)
            loss = utils.EntropyLoss(y, prob)  - regularizer  * tf.norm(perturb)**2

        gradient = g.gradient(loss, x)
        x = x + learning_rate * gradient
    return_loss = utils.EntropyLoss(y, graph(x)) / utils.EntropyLoss(y, graph(x_start))
    
    return return_loss.numpy()

if __name__ == '__main__':


    start, end = int(float(sys.argv[1])), int(float(sys.argv[2]))
    seed_data, seed_model = int(float(sys.argv[3])), int(float(sys.argv[4]))
    lr = float(sys.argv[5])

    dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed_data)

    x_unprotected_train, x_protected_train = dataset_orig_train.features[:, :39], dataset_orig_train.features[:, 39:]
    x_unprotected_test, x_protected_test = dataset_orig_test.features[:, :39], dataset_orig_test.features[:, 39:]
    y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))





    sensetive_directions = []
    protected_regression = linear_model.LogisticRegression(fit_intercept = True)
    protected_regression.fit(x_unprotected_test, x_protected_test[:, 0])
    sensetive_directions.append(protected_regression.coef_.reshape((-1,)))
    protected_regression.fit(x_unprotected_test, x_protected_test[:, 1])
    sensetive_directions.append(protected_regression.coef_.reshape((-1,)))
    sensetive_directions = np.array(sensetive_directions)

    sensetive_directions = scipy.linalg.orth(sensetive_directions.T).T
    for i, s in enumerate(sensetive_directions):
        while np.linalg.norm(s) != 1:
            s = s/ np.linalg.norm(s)
        sensetive_directions[i] = s


    y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
    x_unprotected_train, x_unprotected_test = tf.cast(x_unprotected_train, dtype = tf.float32), tf.cast(x_unprotected_test, dtype = tf.float32)
    y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)
    sensetive_directions = tf.cast(sensetive_directions, dtype = tf.float32)


    with open(f'./sensr/models/data_{seed_data}_{seed_model}.txt', 'r') as f:
        weight = json.load(f)

    weights = [np.array(w) for w in weight]

    def graph(x):
        layer1 = SimpleDense([weights[0], weights[1]])
        layer2 = SimpleDense([weights[2], weights[3]])
        out = tf.nn.relu(layer1(x))
        out = layer2(out)
        prob = tf.nn.softmax(out)
        return prob






    perturbed_test_samples = []
    for data in zip(x_unprotected_test, y_test):
        perturbed_test_samples.append(sample_perturbation(data, regularizer=50, learning_rate=lr, num_steps=200))

    perturbed_test_samples = np.array(perturbed_test_samples)



    filename = f'./sensr/outcome/perturbed_ratio_seed_{seed_data}_{seed_model}_lr_{lr}.npy'


    np.save(filename, perturbed_test_samples)





