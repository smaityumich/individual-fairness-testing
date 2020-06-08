import json
import tensorflow as tf
from tensorflow import keras
import numpy as np

with open('data.txt', 'r') as f:
    weight = json.load(f)

weights = [np.array(w) for w in weight]



def SimpleDense(variable):
    w, b = variable
    w = tf.cast(w, dtype = tf.float32)
    b = tf.cast(b, dtype = tf.float32)
    return lambda x: tf.matmul(x, w) + b

def model(x):
    layer1 = SimpleDense([weights[0], weights[1]])
    layer2 = SimpleDense([weights[2], weights[3]])
    out = tf.nn.relu(layer1(x))
    out = layer2(out)
    prob = tf.nn.softmax(out)
    return prob