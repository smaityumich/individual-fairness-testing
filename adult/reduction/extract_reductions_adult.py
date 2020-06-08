import numpy as np
from adult_data import preprocess_adult_data
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity, TruePositiveRateDifference, ErrorRateRatio, EqualizedOdds
from metrics import group_metrics
constraints = {'TPRD': TruePositiveRateDifference,
               'ERR': ErrorRateRatio,
               'DP': DemographicParity,
               'EO': EqualizedOdds}

import json
np.random.seed(1)
# Adult data processing
seeds = np.load('../seeds.npy')
for i in range(10):
    data_seed = seeds[i, 0]
    print(f'Running data seed {data_seed}')
    dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = data_seed)

    all_train, all_test = dataset_orig_train.features, dataset_orig_test.features
    y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))
    y_train, y_test = y_train.astype('int32'), y_test.astype('int32')

    x_train = np.delete(all_train, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']], axis = 1)
    x_test = np.delete(all_test, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']], axis = 1)

    group_train = dataset_orig_train.features[:, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']]]
    group_test = dataset_orig_test.features[:, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male', 'race_ White']]]
    group_train_cross = group_train[:,0] + group_train[:,1]*2
    group_test_cross = group_test[:,0] + group_test[:,1]*2

    ## Train reductions
    eps = 0.03
    c = 'EO'
    constraint = constraints[c]()
    classifier = LogisticRegression(solver='liblinear', fit_intercept=True, class_weight='balanced')
    mitigator = ExponentiatedGradient(classifier, constraint, eps=eps)
    mitigator.fit(x_train, y_train, sensitive_features=group_train_cross)
    y_pred_mitigated = mitigator.predict(x_test)
    print('\nFair on all test')
    _ = group_metrics(y_test, y_pred_mitigated, group_test[:,0], label_protected=0, label_good=1)


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
