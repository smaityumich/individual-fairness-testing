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
import os

def sample_perturbation(data_point, graph, regularizer = 20, learning_rate = 3e-2, num_steps = 200):
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
    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x)
            prob = graph(x)
            perturb = utils.unprotected_direction(x-x_start, sensetive_directions)
            loss = utils.EntropyLoss(y, prob)  - regularizer * tf.norm(perturb)**2

        gradient = g.gradient(loss, x)
        x = x + learning_rate * gradient
    
    prob = graph(x)
    prob_start = graph(x_start)
    loss_end, loss_start = 1 - utils._accuracy(y, prob), 1 - utils._accuracy(y, prob)
    return_ratio = utils.EntropyLoss(y, prob) / utils.EntropyLoss(y, prob_start)
    return_ratio = return_ratio.numpy()
    loss_end, loss_start = loss_end.numpy(), loss_start.numpy()
    return return_ratio, loss_start, loss_end

if __name__ == '__main__':

    start, end = int(float(sys.argv[5])), int(float(sys.argv[6]))
    seed_data = int(float(sys.argv[1]))
    seed_model = int(float(sys.argv[2]))
    lr = float(sys.argv[3])
    iters = int(float(sys.argv[4]))
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



    graph = tf.keras.models.load_model(f'./project/graphs/graph_{seed_data}_{seed_model}')     


    perturbed_test_samples = []
    for i, data in enumerate(zip(x_unprotected_test[start:end], y_test[start:end])):
        perturbed_test_samples.append(sample_perturbation(data, graph, regularizer=50, learning_rate=lr, num_steps=iters))
        if i % 20 == 0:
            print(f'Done {i}\n\n')

    perturbed_test_samples = np.array(perturbed_test_samples)

    if not os.path.isdir('./project/outcome'):
        os.mkdir('./project/outcome')

    filename = f'./project/outcome/perturbed_ratio_seed_{seed_data}_{seed_model}_lr_{lr}_step_{iters}_start_{start}_end_{end}.npy'


    np.save(filename, perturbed_test_samples)




