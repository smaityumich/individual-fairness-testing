import numpy as np
import tensorflow as tf
from data_preprocess import get_data
from sklearn import linear_model
import utils
import time
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import scipy
plt.ioff()
import sys

def sample_perturbation(data_point, regularizer = 20, learning_rate = 3e-2, num_steps = 200):
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
            loss = utils.EntropyLoss(y, prob)  - regularizer * tf.norm(perturb)**2

        gradient = g.gradient(loss, x)
        x = x + learning_rate * gradient/(i ** (2/3))

    return_loss = utils.EntropyLoss(y, graph(x)) / utils.EntropyLoss(y, graph(x_start))
    print('done')
    return return_loss.numpy()

if __name__ == '__main__':

    start, end = int(float(sys.argv[1])), int(float(sys.argv[2]))
    seed_data = int(float(sys.argv[3]))
    seed_model = int(float(sys.argv[4]))
    lr = float(sys.argv[5])
    x_train, x_test, y_train, y_test, _, y_sex_test, y_race_test = get_data(seed_data)




    sensetive_directions = []
    protected_regression = linear_model.LogisticRegression(fit_intercept = True)
    protected_regression.fit(x_test.numpy(), y_sex_test)
    sensetive_directions.append(protected_regression.coef_.reshape((-1,)))
    protected_regression.fit(x_test.numpy(), y_race_test)
    sensetive_directions.append(protected_regression.coef_.reshape((-1,)))
    sensetive_directions = np.array(sensetive_directions)

    sensetive_directions = scipy.linalg.orth(sensetive_directions.T).T
    for i, s in enumerate(sensetive_directions):
        while np.linalg.norm(s) != 1:
            s = s/ np.linalg.norm(s)
        sensetive_directions[i] = s
    sensetive_directions = tf.cast(sensetive_directions, dtype = tf.float32)



    graph = tf.keras.models.load_model(f'./project/graphs/graph_{seed_data}_{seed_model}')     


    perturbed_test_samples = []
    for data in zip(x_test[start:end], y_test[start:end]):
        perturbed_test_samples.append(sample_perturbation(data, regularizer=50,\
             learning_rate=lr, num_steps=200))

    perturbed_test_samples = np.array(perturbed_test_samples)



    filename = f'./project/outcome/perturbed_ratio_{start}_to_{end}_seed_{seed_data}_{seed_model}_lr_{lr}.npy'


    np.save(filename, perturbed_test_samples)




