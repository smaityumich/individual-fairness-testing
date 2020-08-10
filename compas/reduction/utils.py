import tensorflow as tf



def EntropyLoss(y, prob):
    return -2*tf.reduce_mean(tf.math.multiply(y, tf.math.log(prob)))




def _accuracy(y, ypred):
    acc = tf.cast(tf.equal(y, ypred), dtype = tf.float32)
    return tf.reduce_mean(acc)



def unprotected_direction(x, sensetive_directions):
    x = x - x @ tf.linalg.matrix_transpose(sensetive_directions) @ sensetive_directions
    return x




