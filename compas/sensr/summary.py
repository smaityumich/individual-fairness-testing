import numpy as np
import tensorflow as tf
from compas_data import get_compas_train_test
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


def SimpleDense(variable):
    w, b = variable
    w = tf.cast(w, dtype = tf.float32)
    b = tf.cast(b, dtype = tf.float32)
    return lambda x: tf.matmul(x, w) + b


if __name__ == '__main__':
     seed_data, seed_model = int(float(sys.argv[1])), int(float(sys.argv[2]))
     lr = float(sys.argv[3])

     x_train, x_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train,\
          y_race_test, feature_names = get_compas_train_test(random_state = seed_data)


     with open(f'./sensr/models/data_{seed_data}_{seed_model}.txt', 'r') as f:
        weight = json.load(f)

     weights = [np.array(w) for w in weight]

     def graph(x):
          layer1 = SimpleDense([weights[0], weights[1]])
          layer2 = SimpleDense([weights[2], weights[3]])
          out = tf.nn.relu(layer1(x))
          out = layer2(out)
          prob = tf.nn.softmax(out)
          return prob


     x_test = tf.cast(x_test, dtype = tf.float32)
     prob = graph(x_test)
     y_pred = tf.argmax(prob, axis = 1)
     y_pred = y_pred.numpy()
     gender = y_sex_test
     race = y_race_test
     #y_test = y_test.numpy()[:, 1]
     
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

     
     filename = f'./sensr/outcome/perturbed_ratio_0_to_1000_seed_{seed_data}_{seed_model}_lr_{lr}.npy'
     a = np.load(filename)
     a = a[np.isfinite(a)]
     lb = np.mean(a) - 1.645*np.std(a)/np.sqrt(a.shape[0])
     t = (np.mean(a)-1.25)/np.std(a)
     t *= np.sqrt(a.shape[0])
     pval = 1- norm.cdf(t)

     save_dict = {'algo': 'sensr', 'seed': (seed_data, seed_model), 'lr': lr, 'accuracy': accuracy}
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
