import numpy as np
from compas_data import get_compas_train_test
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds
from metrics import group_metrics
#from utils import sample_balanced, reductions_prob

import json
np.random.seed(1)
# Adult data processing
seeds = np.load('../seeds.npy')
for i in range(1):
    data_seed = seeds[i, 0]
    print(f'Running data seed {data_seed}')
    x_train,x_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test,\
         feature_names = get_compas_train_test(random_state = data_seed)
    #x_train, x_test = x_train[:, 2:], x_test[:, 2:]

    group_train_cross = y_sex_train + 2*y_race_train
    group_test_cross = y_sex_test + 2*y_race_test



    #### Using 4 protected attributes ####
    ## Reduction classifier
    eps = 0.05
    constraint = EqualizedOdds()
    classifier = LogisticRegression(solver='liblinear', fit_intercept=True)
    mitigator = ExponentiatedGradient(classifier, constraint, eps=eps, T=50)
    mitigator.fit(x_train, y_train, sensitive_features=group_train_cross)
    y_pred_mitigated = mitigator.predict(x_test)
    print('\nFair on all test')
    _ = group_metrics(y_test, y_pred_mitigated, y_race_test, label_protected=0, label_good=0)

    ens_weights = []
    coefs = []
    intercepts = []

    for t, w_t in enumerate(mitigator._weights.index):
        if mitigator._weights[w_t] > 0:
            coefs.append(mitigator._predictors[t].coef_.flatten())
            intercepts.append(mitigator._predictors[t].intercept_[0])
            ens_weights.append(mitigator._weights[w_t])

    ens_weight = [e.tolist() for e in ens_weights]
    coef = [c.tolist() for c in coefs]
    intercept = [i.tolist() for i in intercepts]

    data = {'ens_weights': ens_weight, 'coefs': coef, 'intercepts': intercept}
    with open(f'models/data_{data_seed}.txt', 'w') as f:
        json.dump(data, f)
