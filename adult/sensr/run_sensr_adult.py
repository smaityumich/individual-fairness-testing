import numpy as np
from adult_data import preprocess_adult_data
from sklearn.linear_model import LogisticRegression
from metrics import group_metrics
from sklearn.preprocessing import OneHotEncoder
from train_clp_adult import train_fair_nn
import tensorflow.compat.v1 as tf
import json
import sys


# np.save('seeds.npy', seeds)
def run_sensr(seed_data, seed_model):
    

    #seed_data = int(float(sys.argv[1]))
    #seed_model = int(float(sys.argv[2]))

    dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed_data)

    all_train, all_test = dataset_orig_train.features, dataset_orig_test.features
    y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))
    y_train, y_test = y_train.astype('int32'), y_test.astype('int32')

    x_train = np.delete(all_train, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']], axis = 1)
    x_test = np.delete(all_test, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']], axis = 1)

    group_train = dataset_orig_train.features[:, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']]]
    group_test = dataset_orig_test.features[:, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']]]
    group_names = ['Gender', 'Race']

    one_hot = OneHotEncoder(sparse=False)
    one_hot.fit(y_train.reshape(-1,1))
    names_income = one_hot.categories_
    y_train = one_hot.transform(y_train.reshape(-1,1))
    y_test = one_hot.transform(y_test.reshape(-1,1))

    sensitive_directions = []
    for y_protected in group_train.T:
    	lr = LogisticRegression(solver='liblinear', fit_intercept=True)
    	lr.fit(x_train, y_protected)
    	sensitive_directions.append(lr.coef_.flatten())

    sensitive_directions = np.array(sensitive_directions)

    tf.reset_default_graph()
    fair_info = [group_train, group_test, group_names, sensitive_directions]
    weights, train_logits, test_logits, _, variables = train_fair_nn(x_train, y_train, tf_prefix='sensr', adv_epoch_full=50, l2_attack=0.0001,
                                          adv_epoch=10, ro=0.001, adv_step=10., plot=True, fair_info=fair_info, balance_batch=True, 
                                          X_test = x_test, X_test_counter=None, y_test = y_test, lamb_init=2., 
                                          n_units=[100], l2_reg=0, epoch=20000, batch_size=1000, lr=1e-5, lambda_clp=0.,
                                          fair_start=0., counter_init=False, seed=None)

    print('Gender:')
    _ = group_metrics(y_test[:,1], test_logits.argmax(axis=1), group_test[:,0], label_protected=0, label_good=1)
    print('\nRace:')
    _ = group_metrics(y_test[:,1], test_logits.argmax(axis=1), group_test[:,1], label_protected=0, label_good=1)


    weight = [w.tolist() for w in weights]
    with open(f'models/data_{seed_data}_{seed_model}.txt', 'w') as f:
        json.dump(weight, f)

if __name__ == "__main__":
    np.random.seed(1)
    seeds = np.load('../seeds.npy')
    i = int(sys.argv[1])
    seed_data = seeds[i, 0]
    seed_model = seeds[i, 1]
    run_sensr(seed_data, seed_model)
