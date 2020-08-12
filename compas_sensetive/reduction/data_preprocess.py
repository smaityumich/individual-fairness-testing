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
        y_race_train, y_race_test, _ = compas.get_compas_train_test()
    x_train, x_test = x_train[:, 2:], x_test[:, 2:]

    # Standardizing the last four columns
    for i in range(1, 5):
        x_train[:, i] = standardize(x_train[:, i])
        x_test[:, i] = standardize(x_test[:, i])

    # Calculate the sensitive directions
    sensetive_directions = []
    protected_regression = linear_model.LogisticRegression(fit_intercept = True)
    protected_regression.fit(x_train, y_sex_train)
    sensetive_directions.append(protected_regression.coef_.reshape((-1,)))
    protected_regression.fit(x_train, y_race_train)
    sensetive_directions.append(protected_regression.coef_.reshape((-1,)))
    sensetive_directions = np.array(sensetive_directions)

    # Extrancting orthornormal basis for sensitive directions
    sensetive_directions = scipy.linalg.orth(sensetive_directions.T).T
    for i, s in enumerate(sensetive_directions):
        while np.linalg.norm(s) != 1:
            s = s/ np.linalg.norm(s)
        sensetive_directions[i] = s

    # casting to tensor
    x_train, x_test = tf.cast(x_train, dtype=tf.float32), tf.cast(x_test, dtype = tf.float32)
    sensetive_directions = tf.cast(sensetive_directions, dtype=tf.float32)
    y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
    y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)

    return x_train, x_test, y_train, y_test, sensetive_directions, y_sex_test, y_race_test