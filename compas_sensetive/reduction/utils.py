import tensorflow as tf
from sklearn import linear_model
import scipy
import numpy as np



def EntropyLoss(y, prob):
    return -2*tf.reduce_mean(tf.math.multiply(y, tf.math.log(prob)))




def _accuracy(y, ypred):
    acc = tf.cast(tf.equal(y, ypred), dtype = tf.float32)
    return tf.reduce_mean(acc)



def unprotected_direction(x, sensetive_directions):
    x = x - x @ tf.linalg.matrix_transpose(sensetive_directions) @ sensetive_directions
    return x


def sensitive_dir(x, gender, race):
    d = x.shape[1]
    sensetive_directions = []
    protected_regression = linear_model.LogisticRegression(fit_intercept = True)
    protected_regression.fit(x[:, 2:], gender)
    a = protected_regression.coef_.reshape((-1,))
    a = np.concatenate(([0, 0], a), axis=0)
    sensetive_directions.append(a)
    protected_regression.fit(x[:,2:], race)
    a = protected_regression.coef_.reshape((-1,))
    a = np.concatenate(([0, 0], a), axis=0)
    sensetive_directions.append(a)
    a, b = np.zeros((d,)), np.zeros((d,))
    a[0], b[1] = 1, 1
    sensetive_directions.append(a)
    sensetive_directions.append(b)
    sensetive_directions = np.array(sensetive_directions)

    # Extrancting orthornormal basis for sensitive directions
    sensetive_basis = scipy.linalg.orth(sensetive_directions.T).T
    for i, s in enumerate(sensetive_basis):
        #while np.linalg.norm(s) != 1:
        s = s/ np.linalg.norm(s)
        sensetive_basis[i] = s

    return sensetive_directions, sensetive_basis


