import numpy as np
import tensorflow as tf
from adult_modified import preprocess_adult_data
from sklearn import linear_model
import utils
import time
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import scipy
import metrics
plt.ioff()
import sys
import json
from scipy.stats import norm

if __name__ == '__main__':
     seed_data, seed_model = int(float(sys.argv[1])), int(float(sys.argv[2]))
     lr = float(sys.argv[3])

     dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed_data)

     x_unprotected_train, x_protected_train = dataset_orig_train.features[:, :39], dataset_orig_train.features[:, 39:]
     x_unprotected_test, x_protected_test = dataset_orig_test.features[:, :39], dataset_orig_test.features[:, 39:]
     y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))


# Casing to tensor 
#y_train, y_test = y_train.astype('int32'), y_test.astype('int32')
     x_unprotected_train, x_unprotected_test = tf.cast(x_unprotected_train, dtype = tf.float32), tf.cast(x_unprotected_test, dtype = tf.float32)
#y_train, y_test = tf.one_hot(y_train, 2), tf.one_hot(y_test, 2)

     graph = tf.keras.models.load_model(f'./baseline_bal/graphs/graph_{seed_data}_{seed_model}')


 
     prob = graph(x_unprotected_test)
     y_pred = tf.argmax(prob, axis = 1)
     y_pred = y_pred.numpy()
     gender = dataset_orig_test.features[:, 39]
     race = dataset_orig_test.features[:, 40]
     
     print('\n\nMeasures for gender\n')
     accuracy, bal_acc, \
            gap_rms_gen, mean_gap_gen, max_gap_gen, \
            average_odds_difference_gen, equal_opportunity_difference_gen,\
                 statistical_parity_difference_gen = metrics.group_metrics(y_test, y_pred, gender, label_good=1)

     print('\n\n\nMeasures for race\n')
     accuracy, bal_acc, \
            gap_rms_race, mean_gap_race, max_gap_race, \
            average_odds_difference_race, equal_opportunity_difference_race,\
                 statistical_parity_difference_race = metrics.group_metrics(y_test, y_pred, race, label_good=1)

     
     filename = f'./baseline_bal/outcome/perturbed_ratio_seed_{seed_data}_{seed_model}_lr_{lr}.npy'
     a = np.load(filename)

     a = a[np.isfinite(a)]
     lb = np.mean(a) - 1.645*np.std(a)/np.sqrt(a.shape[0])
     t = (np.mean(a)-1.25)/np.std(a)
     t *= np.sqrt(a.shape[0])
     pval = 1- norm.cdf(t)

     save_dict = {'algo': 'baseline_bal', 'seed': (seed_data, seed_model), 'lr': lr, 'accuracy': accuracy}
     save_dict['lb'] = lb
     save_dict['pval'] = pval
     save_dict['bal_acc'], \
            save_dict['gap_rms_gen'], save_dict['mean_gap_gen'], save_dict['max_gap_gen'], \
            save_dict['average_odds_difference_gen'], save_dict['equal_opportunity_difference_gen'],\
                 save_dict['statistical_parity_difference_gen'] = bal_acc, \
            gap_rms_gen, mean_gap_gen, max_gap_gen, \
            average_odds_difference_gen, equal_opportunity_difference_gen,\
                 statistical_parity_difference_gen

     save_dict['bal_acc'], \
            save_dict['gap_rms_race'], save_dict['mean_gap_race'], save_dict['max_gap_race'], \
            save_dict['average_odds_difference_race'], save_dict['equal_opportunity_difference_race'],\
                 save_dict['statistical_parity_difference_race'] = bal_acc, \
            gap_rms_race, mean_gap_race, max_gap_race, \
            average_odds_difference_race, equal_opportunity_difference_race,\
                 statistical_parity_difference_race

     with open('all_summary.out', 'a') as f:
          f.writelines(str(save_dict) + '\n')