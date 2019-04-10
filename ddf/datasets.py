import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pdb
import re
from collections import OrderedDict
def _filter_features_by_prefixes(df, prefixes):
    res = []
    for name in df.columns:
        filtered = False
        for pref in prefixes:
            if name.startswith(pref):
                filtered = True
                break
        if not filtered:
            res.append(name)
    return res


def _split_feature_names(df, data_name, feature_split):
    if data_name == "adult":
        if feature_split == "sex_salary":
            x_features = _filter_features_by_prefixes(df, ['sex', 'salary'])
            s_features = ['sex_Male']
            y_features = ['salary_>50K']
        elif feature_split == "race_salary":
            x_features = _filter_features_by_prefixes(
                df, ['race', 'salary'])
            s_features = [
                    'race_Amer-Indian-Eskimo',
                    'race_Asian-Pac-Islander',
                    'race_Black',
                    'race_White',
                ]
            y_features = ['salary_>50K']
        elif feature_split == "sex-race_salary":
            x_features = _filter_features_by_prefixes(
                df, ['sex', 'race', 'salary'])
            s_features = [
                    'sex_Male',
                    'race_Amer-Indian-Eskimo',
                    'race_Asian-Pac-Islander',
                    'race_Black',
                    'race_White',
                ]
            y_features = ['salary_>50K']

        else:
            raise NotImplementedError()
    elif data_name== "nypd":
        if feature_split== "sex_possession":
            x_features = _filter_features_by_prefixes(df, ['sex', 'possession'])
            s_features = ['sex_M']
            y_features = ['possession']
        elif feature_split== "sex-race_possession":
            x_features = _filter_features_by_prefixes(
                df, ['sex', 'race', 'possession'])
            s_features = [
                'sex_M',
                'sex_F',
                'sex_Z',
                'race_A',
                'race_B',
                'race_I',
                'race_P',
                'race_Q',
                'race_U',
                'race_W',
                'race_Z',
            ]
            y_features = ['possession']
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return x_features, s_features, y_features


def _split_features(df, data_name, feature_split):

    # TODO: I've added this line to make so only dealing with white and black
    # df = df.query('race_Black' != 'race_White')

    x_features, s_features, y_features = _split_feature_names(
        df, data_name, feature_split)

    #allowing to go back from one hot encoded features to categorical features
    if data_name=="adult":
        SORTED_FEATURES_NAMES = [
                'age',
                'education-num',
                'capital-gain',
                'capital-loss',
                'hours-per-week',
                'workclass',
                'education',
                'marital-status',
                'occupation',
                'relationship',
                'race',
                'sex',
                'native-country',
                'salary'   
            ]

    features = OrderedDict()
    for i in range(len(SORTED_FEATURES_NAMES)):
        features[SORTED_FEATURES_NAMES[i]] = [not re.match(SORTED_FEATURES_NAMES[i],values)==None for values in x_features]

    #fixing the education to not count education-num
    features['education'][1] = False
    x = df[x_features].values.astype(float)
    s = df[s_features].values.astype(float)
    y = df[y_features].values.astype(float)

    return x, s, y, features


def scale(scaler_class, train, valid, test):
    if scaler_class is None:
        return [train, valid, test]

    scaler = scaler_class()
    #train_scaled = scaler.fit_transform(train)
    #valid_scaled = scaler.fit_transform(valid)
    #test_scaled = scaler.fit_transform(test)
    scalerobj = scaler.fit(np.concatenate((np.concatenate((train,valid),axis=0),test),axis=0))
    train_scaled = scalerobj.transform(train)
    valid_scaled = scalerobj.transform(valid)
    test_scaled = scalerobj.transform(test)
    return [train_scaled, valid_scaled, test_scaled]


def load_dataset(train_path, test_path, validation_size, random_state,
                 data_name, feature_split, remake_test=False, test_size=None,
                 input_scaler=StandardScaler, sensitive_scaler=None):
    df_train_raw = pd.read_csv(train_path, engine='c')
    df_test_raw = pd.read_csv(test_path)

    if remake_test:
        if test_size is None:
            test_size = df_test_raw.shape[0]

        df_full = pd.concat([df_train_raw, df_test_raw])
        df_full_shuffled = df_full.sample(frac=1, random_state=random_state)

        df_train_valid = df_full_shuffled[:-test_size]
        df_test = df_full_shuffled[-test_size:]

    else:
        if test_size is not None:
            raise ValueError("Changing test size is only possible "
                             "if remake_test is True.")

        df_train_valid = df_train_raw.sample(frac=1, random_state=random_state)
        df_test = df_test_raw

    df_train = df_train_valid[:-validation_size]
    df_valid = df_train_valid[-validation_size:]

    x_train, s_train, y_train, cat_features = _split_features(df_train, data_name, 
                                                feature_split)
    x_valid, s_valid, y_valid,_ = _split_features(df_valid, data_name, 
                                                feature_split)
    x_test, s_test, y_test,_ = _split_features(df_test, data_name, 
                                             feature_split)

    x_train, x_valid, x_test = scale(
        input_scaler, x_train, x_valid, x_test)

    s_train, s_valid, s_test = scale(
        sensitive_scaler, s_train, s_valid, s_test)

    data_train = x_train, s_train, y_train
    data_valid = x_valid, s_valid, y_valid
    data_test = x_test, s_test, y_test

    return data_train, data_valid, data_test, cat_features
