import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import logging
import seaborn as sns
from log_code import setup_logging
logger = setup_logging('cat_to_num')
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder


def changing_data_to_num(X_train_cat,X_test_cat):
    try:
        one_hot = OneHotEncoder(drop='first')
        # since gender and region columns are nominal we are going to apply OnehotEnoder
        one_hot.fit(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling',  'Sim_Providers']])
        result = one_hot.transform(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'Sim_Providers']]).toarray()
        result1 = one_hot.transform(X_test_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling',  'Sim_Providers']]).toarray()
        f = pd.DataFrame(data=result, columns=one_hot.get_feature_names_out())
        f1 = pd.DataFrame(data=result1, columns=one_hot.get_feature_names_out())
        X_train_cat.reset_index(drop=True, inplace=True)
        f.reset_index(drop=True, inplace=True)
        X_test_cat.reset_index(drop=True, inplace=True)
        f1.reset_index(drop=True, inplace=True)
        X_train_cat = pd.concat([X_train_cat, f], axis=1)
        X_train_cat = X_train_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'Sim_Providers'], axis=1)
        X_test_cat = pd.concat([X_test_cat, f1], axis=1)
        X_test_cat = X_test_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'Sim_Providers'], axis=1)
        logger.info(f"{X_train_cat.columns}")
        logger.info(f"{X_train_cat.sample(10)}")
        logger.info(f"{X_test_cat.columns}")
        logger.info(f"{X_test_cat.sample(10)}")
        logger.info('================Odinal Encoding==========================')
        od = OrdinalEncoder()
        od.fit(X_train_cat[['InternetService', 'Contract', 'PaymentMethod']])
        r1 = od.transform(X_train_cat[['InternetService', 'Contract', 'PaymentMethod']])
        r2 = od.transform(X_test_cat[['InternetService', 'Contract', 'PaymentMethod']])
        c_names = od.get_feature_names_out()
        c_names = c_names + '_Od'
        g1 = pd.DataFrame(data=r1, columns=c_names)
        g2 = pd.DataFrame(data=r2, columns=c_names)
        X_train_cat.reset_index(drop=True, inplace=True)
        g1.reset_index(drop=True, inplace=True)
        X_train_cat = pd.concat([X_train_cat, g1], axis=1)
        X_train_cat = X_train_cat.drop(['InternetService', 'Contract', 'PaymentMethod'], axis=1)

        X_test_cat.reset_index(drop=True, inplace=True)
        g2.reset_index(drop=True, inplace=True)
        X_test_cat = pd.concat([X_test_cat, g2], axis=1)
        X_test_cat = X_test_cat.drop(['InternetService', 'Contract', 'PaymentMethod'], axis=1)

        logger.info(f"{X_train_cat.columns}")
        logger.info(f"{X_train_cat.sample(10)}")
        logger.info(f"{X_test_cat.columns}")
        logger.info(f"{X_test_cat.sample(10)}")

        logger.info(f"{X_train_cat.isnull().sum()}")
        logger.info(f"{X_test_cat.isnull().sum()}")

        return X_train_cat,X_test_cat

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")