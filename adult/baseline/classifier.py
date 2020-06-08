import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import utils









def Classifier(graph, x_train, y_train,  num_steps = 10000, batch_size = 125, learning_rate = 1e-4, seed = 1):
    # Tensor slice for train data
    # Partition train data
    tf.random.set_seed(seed)
    index0, index1 = y_train[:, 1] == 0 , y_train[:, 1] == 1
    x_train0, x_train1 = x_train[index0], x_train[index1]
    y_train0, y_train1 = y_train[index0], y_train[index1]

    batch = tf.data.Dataset.from_tensor_slices((x_train0, y_train0))
    batch = batch.repeat().shuffle(5000).batch(batch_size)
    batch_data0= batch.take(num_steps)

    batch = tf.data.Dataset.from_tensor_slices((x_train1, y_train1))
    batch = batch.repeat().shuffle(5000).batch(batch_size)
    batch_data1= batch.take(num_steps)

    

    # Adam optimizer
    optimizer = tf.optimizers.Adam(learning_rate)

    def train_step(data_train_epoch, step):
        x, y = data_train_epoch
        with tf.GradientTape() as g:
            loss = utils.EntropyLoss(y, graph(x, predict = False))

        variables = graph.trainable_variables
        gradients = g.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    
    for step, (data0, data1) in enumerate(zip(batch_data0, batch_data1), 1):
        x0, y0 = data0
        x1, y1 = data1
        x = tf.concat([x0, x1], axis = 0)
        y = tf.concat([y0, y1], axis = 0)
        batch_data_train = x, y
        train_step(batch_data_train, step)
        if step % 200 == 0:
            print(f'Done step {step}\n')
    
    return graph

