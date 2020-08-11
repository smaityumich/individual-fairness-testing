import numpy as np
import tensorflow as tf
import classifier as cl
import utils
from tensorflow import keras
from data_preprocess import get_data



seeds = np.load('../seeds.npy')


for i in range(10):
    data_seed = seeds[i, 0]
    expt_seed = seeds[i, 1]
    x_train, x_test, y_train, y_test, sensetive_directions, y_sex_test, y_race_test \
    = get_data(random_state=data_seed)



    print(f'Running data seed {data_seed} and expt seed {expt_seed}')
    init_graph = utils.ClassifierGraph([50,], 2, sensetive_directions = sensetive_directions,\
     input_shape=(5, ), seed_model=expt_seed)
    graph = cl.Classifier(init_graph, x_train, y_train, num_steps = 8000, seed = expt_seed) # use for unfair algo
    #graph.model._set_inputs((-1, 5))
    inputs = keras.Input((5,))

    outputs = graph(inputs)
    model = keras.Model(inputs, outputs)
    model.save(f'graphs/graph_{data_seed}_{expt_seed}')
    #graph.model.save(f'graphs/graph_{data_seed}_{expt_seed}')
