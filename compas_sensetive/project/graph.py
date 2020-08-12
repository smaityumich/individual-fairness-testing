import numpy as np
import tensorflow as tf
import classifier as cl
import utils
from tensorflow import keras
from data_preprocess import get_data
from compas_data import get_compas_train_test



seeds = np.load('../seeds.npy')


for i in range(1):
    data_seed = seeds[i, 0]
    expt_seed = seeds[i, 1]
    x_train, x_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train,\
          y_race_test, feature_names = get_compas_train_test(random_state = data_seed)

    y_sex_train, y_sex_test, y_race_train, y_race_test = np.copy(y_sex_train), np.copy(y_sex_test),\
        np.copy(y_race_train), np.copy(y_race_test)

    _, sensetive_directions = utils.sensitive_dir(x_train, y_sex_train, y_race_train)
    sensetive_directions = tf.cast(sensetive_directions, dtype = tf.float32)
     

    x_train = tf.cast(x_train, dtype = tf.float32)
    y_train = y_train.astype('int32')
    y_train = tf.one_hot(y_train, 2)



    print(f'Running data seed {data_seed} and expt seed {expt_seed}')
    init_graph = utils.ClassifierGraph([50,], 2, sensetive_directions = sensetive_directions,\
     input_shape=(7, ), seed_model=expt_seed)
    graph = cl.Classifier(init_graph, x_train, y_train, num_steps = 8000, seed = expt_seed) # use for unfair algo
    #graph.model._set_inputs((-1, 5))
    inputs = keras.Input((7,))

    outputs = graph(inputs)
    model = keras.Model(inputs, outputs)
    model.save(f'graphs/graph_{data_seed}_{expt_seed}')
    #graph.model.save(f'graphs/graph_{data_seed}_{expt_seed}')
