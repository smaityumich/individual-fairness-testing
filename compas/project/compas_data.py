from aif360.datasets import CompasDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_preproc_data_compas(protected_attributes=None):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
        """

        df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text',
                 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score',
                 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

        # Indices of data samples to keep
        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix
        ix = (df['is_recid'] != -1) & ix
        ix = (df['c_charge_degree'] != "O") & ix
        ix = (df['score_text'] != 'N/A') & ix
        df = df.loc[ix,:]
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-
                                pd.to_datetime(df['c_jail_in'])).apply(
                                                        lambda x: x.days)

        # Restrict races to African-American and Caucasian
        dfcut = df.loc[~df['race'].isin(['Native American','Hispanic','Asian','Other']),:]

        # Restrict the features to use
        dfcutQ = dfcut[['sex','race','age_cat','c_charge_degree','score_text','priors_count','is_recid',
                'two_year_recid','length_of_stay']].copy()


        # Quantize length of stay
        def quantizeLOS(x):
            if x<= 7:
                return '<week'
            if 8<x<=93:
                return '<3months'
            else:
                return '>3 months'

        # Quantize length of stay
        def adjustAge(x):
            if x == '25 - 45':
                return '25 to 45'
            else:
                return x

        # Quantize score_text to MediumHigh
        def quantizeScore(x):
            if (x == 'High')| (x == 'Medium'):
                return 'MediumHigh'
            else:
                return x

        def group_race(x):
            if x == "Caucasian":
                return 1.0
            else:
                return 0.0
            
        dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(lambda x: quantizeLOS(x))
        dfcutQ['score_text'] = dfcutQ['score_text'].apply(lambda x: quantizeScore(x))
        dfcutQ['age_cat'] = dfcutQ['age_cat'].apply(lambda x: adjustAge(x))

        # Recode sex and race
        dfcutQ['sex'] = dfcutQ['sex'].replace({'Female': 1.0, 'Male': 0.0})
        dfcutQ['race'] = dfcutQ['race'].apply(lambda x: group_race(x))
        features = ['two_year_recid',
                    'sex', 'race',
                    'age_cat', 'priors_count', 'c_charge_degree']

        # Pass vallue to df
        df = dfcutQ[features]

        return df

    XD_features = ['age_cat', 'c_charge_degree', 'priors_count', 'sex', 'race']
    D_features = ['sex', 'race']  if protected_attributes is None else protected_attributes
    Y_features = ['two_year_recid']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['age_cat', 'c_charge_degree']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {0.0: 'Male', 1.0: 'Female'},
                                    "race": {1.0: 'Caucasian', 0.0: 'Not Caucasian'}}


    return CompasDataset(
        label_name=Y_features[0],
        favorable_classes=[0],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=[],
        metadata={'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

def get_compas_orig():
    dataset_orig = load_preproc_data_compas()
    print(dataset_orig.features.shape)
    print(dataset_orig.feature_names)
    dataset_orig.features = dataset_orig.features[:,:-1]

    return dataset_orig


def get_compas_train_test(pct=0.8, random_state = 0):
    
    dataset_orig = get_compas_orig()
    
    # we will standardize continous features
    continous_features = [
            'priors_count'
        ]
    continous_features_indices = [
            dataset_orig.feature_names.index(feat) 
            for feat in continous_features
        ]
    
    # Get the dataset and split into train and test
    # dataset_orig_train, dataset_orig_test = dataset_orig.split([pct], shuffle=True)
    X, y = dataset_orig.features, dataset_orig.labels
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=pct, stratify=y,\
     random_state = random_state)
    
    y_train = np.reshape(y_train, (-1, ))
    y_test = np.reshape(y_test, (-1, ))
    
    sind = dataset_orig.feature_names.index('sex')
    rind = dataset_orig.feature_names.index('race')
    print(dataset_orig.feature_names[sind])
    print(dataset_orig.feature_names[rind])
    y_sex_train = X_train[:, sind]
    y_sex_test = X_test[:, sind]
    y_race_train = X_train[:, rind]
    y_race_test = X_test[:, rind]
    
    ### PROCESS TRAINING DATA
    # normalize continuous features
    SS = StandardScaler().fit(X_train[:, continous_features_indices])
    X_train[:, continous_features_indices] = SS.transform(
            X_train[:, continous_features_indices]
    )
        
    ### PROCESS TEST DATA
    # normalize continuous features
    X_test[:, continous_features_indices] = SS.transform(
            X_test[:, continous_features_indices]
    )

    return X_train, X_test, y_train, y_test, y_sex_train, y_sex_test, y_race_train, y_race_test, dataset_orig.feature_names