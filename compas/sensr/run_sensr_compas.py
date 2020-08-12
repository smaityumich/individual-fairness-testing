import numpy as np
from sklearn.linear_model import LogisticRegression
from metrics import group_metrics
from sklearn.preprocessing import OneHotEncoder
from train_clp_adult import train_fair_nn
import tensorflow.compat.v1 as tf
import json
import sys
from sklearn import linear_model
import compas_data as compas
import utils

def standardize(x):
    return (x - np.mean(x))/np.std(x)

# np.save('seeds.npy', seeds)
def run_sensr(seed_data, seed_model, save_model = True):
    

    # Extracting compas data
    x_train, x_test, y_train, y_test, y_sex_train, y_sex_test,\
        y_race_train, y_race_test, _ = compas.get_compas_train_test(random_state = seed_data)
    group_train, group_test = np.copy(x_train[:, :2]), np.copy(x_test[:, :2])
    y_sex_train, y_sex_test, y_race_train, y_race_test = np.copy(y_sex_train), np.copy(y_sex_test),\
        np.copy(y_race_train), np.copy(y_race_test)
    
    
    group_names = ['sex', 'race']


    one_hot = OneHotEncoder(sparse=False)
    one_hot.fit(y_train.reshape(-1,1))
    names_income = one_hot.categories_
    y_train = one_hot.transform(y_train.reshape(-1,1))
    y_test = one_hot.transform(y_test.reshape(-1,1))
    
    # Standardizing the last four columns
    #for i in range(7):
    #    if i != 2:
    #        x_train[:, i] = standardize(x_train[:, i])
    #        x_test[:, i] = standardize(x_test[:, i])

    

    # Calculate the sensitive directions
    sensetive_directions, _ = utils.sensitive_dir(x_train, y_sex_train, y_race_train)

    tf.reset_default_graph()
    fair_info = [group_train, group_test, group_names, sensetive_directions]
    weights, train_logits, test_logits, _, variables = train_fair_nn(x_train, y_train, tf_prefix='sensr', adv_epoch_full=8, l2_attack=0.0001,
                                          adv_epoch=10, ro=0.001, adv_step=0.1, plot=save_model, fair_info=fair_info, balance_batch=True, 
                                          X_test = x_test, X_test_counter=None, y_test = y_test, lamb_init=2., 
                                          n_units=[100], l2_reg=0, epoch=16000, batch_size=1000, lr=10e-5, lambda_clp=0.,
                                          fair_start=0., counter_init=False, seed=seed_model)

    print('Gender:')
    _ = group_metrics(y_test[:,1], test_logits.argmax(axis=1), group_test[:,0], label_protected=0, label_good=1)
    print('\nRace:')
    _ = group_metrics(y_test[:,1], test_logits.argmax(axis=1), group_test[:,1], label_protected=0, label_good=1)


    weight = [w.tolist() for w in weights]
    with open(f'models/data_{seed_data}_{seed_model}.txt', 'w') as f:
        json.dump(weight, f)
    return None
    

if __name__ == "__main__":
    np.random.seed(1)
    seeds = np.load('../seeds.npy')
    i = int(sys.argv[1])
    seed_data = seeds[i, 0]
    seed_model = seeds[i, 1]
    print(f'Running for {seed_data} {seed_model}\n\n')
    run_sensr(seed_data, seed_model)
