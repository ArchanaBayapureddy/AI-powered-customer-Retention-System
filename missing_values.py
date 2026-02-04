import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from log_code import setup_logging
logger=setup_logging('missing_values')

# ---------------- MEAN ----------------
def mean_imputation(X_train, X_test, col):
    try:
        value = X_train[col].mean()
        X_train[col+'_mean'] = X_train[col].fillna(value)
        X_test[col+'_mean'] = X_test[col].fillna(value)

        logger.info(f"{col}_mean std: {X_train[col+'_mean'].std()} --- original: {X_train[col].std()}")
        return X_train, X_test

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')


# ---------------- MEDIAN ----------------
def median_imputation(X_train, X_test, col):
    try:
        value = X_train[col].median()
        X_train[col+'_median'] = X_train[col].fillna(value)
        X_test[col+'_median'] = X_test[col].fillna(value)

        logger.info(f"{col}_median std: {X_train[col+'_median'].std()} --- original: {X_train[col].std()}")
        return X_train, X_test

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')


# ---------------- MODE ----------------
def mode_imputation(X_train, X_test, col):
    logger.info(f'before null values {X_train.isnull().sum()}')
    logger.info(f'before null values {X_test.isnull().sum()}')
    try:
        value = X_train[col].mode()[0]
        X_train[col+'_mode'] = X_train[col].fillna(value)
        X_test[col+'_mode'] = X_test[col].fillna(value)
        logger.info(f"{col}_mode std: {X_train[col+'_mode'].std()} --- original: {X_train[col].std()}")
        X_train.drop('TotalCharges', axis=1, inplace=True)
        X_test.drop('TotalCharges', axis=1, inplace=True)
        logger.info(f'after null values {X_train.isnull().sum()}')
        logger.info(f'after null values {X_test.isnull().sum()}')
        return X_train, X_test


    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')


# ---------------- CONSTANT ----------------
def constant_imputation(X_train, X_test, col, value=0):
    try:
        X_train[col+'_const'] = X_train[col].fillna(value)
        X_test[col+'_const'] = X_test[col].fillna(value)

        logger.info(f"{col}_const std: {X_train[col+'_const'].std()} --- original: {X_train[col].std()}")
        return X_train, X_test

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')


# ---------------- ARBITRARY ----------------
def arbitrary_imputation(X_train, X_test, col, value=-1):
    try:
        X_train[col+'_arb'] = X_train[col].fillna(value)
        X_test[col+'_arb'] = X_test[col].fillna(value)

        logger.info(f"{col}_arb std: {X_train[col+'_arb'].std()} --- original: {X_train[col].std()}")
        return X_train, X_test

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')


# ---------------- END OF DISTRIBUTION ----------------
def eod_imputation(X_train, X_test, col):
    try:
        eod = X_train[col].mean() + 3 * X_train[col].std()

        X_train[col+'_eod'] = X_train[col].fillna(eod)
        X_test[col+'_eod'] = X_test[col].fillna(eod)

        logger.info(f"{col}_eod std: {X_train[col+'_eod'].std()} --- original: {X_train[col].std()}")
        return X_train, X_test

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')


# ---------------- RANDOM SAMPLE ----------------
def random_sample_imputation(X_train, X_test, col):
    try:
        sample_train = X_train[col].dropna().sample(
            X_train[col].isnull().sum(), replace=True, random_state=42
        )
        sample_train.index = X_train[X_train[col].isnull()].index

        sample_test = X_train[col].dropna().sample(
            X_test[col].isnull().sum(), replace=True, random_state=42
        )
        sample_test.index = X_test[X_test[col].isnull()].index

        X_train[col+'_rsi'] = X_train[col]
        X_train.loc[X_train[col].isnull(), col+'_rsi'] = sample_train

        X_test[col+'_rsi'] = X_test[col]
        X_test.loc[X_test[col].isnull(), col+'_rsi'] = sample_test

        logger.info(f"{col}_rsi std: {X_train[col+'_rsi'].std()} --- original: {X_train[col].std()}")
        return X_train, X_test

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')


# ---------------- FORWARD FILL ----------------
def ffill_imputation(X_train, X_test, col):
    try:
        X_train[col + '_ffill'] = X_train[col].ffill()
        X_test[col + '_ffill'] = X_test[col].ffill()

        logger.info(f"{col}_ffill std: {X_train[col+'_ffill'].std()} --- original: {X_train[col].std()}")
        return X_train, X_test

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')


# ---------------- BACKWARD FILL ----------------
def bfill_imputation(X_train, X_test, col):
    try:

        X_train[col + '_bfill'] = X_train[col].bfill()
        X_test[col + '_bfill'] = X_test[col].bfill()

        logger.info(f"{col}_bfill std: {X_train[col+'_bfill'].std()} --- original: {X_train[col].std()}")
        return X_train, X_test

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')

def find_best_imputation(X_train, X_test, col):
    """
    Compares all imputation methods using standard deviation
    and logs the best method
    """
    try:
        results = {}

        # original std
        original_std = X_train[col].std()
        logger.info(f'Original STD for {col}: {original_std}')

        # ---------- MEAN ----------
        X_tr, X_te = mean_imputation(X_train.copy(), X_test.copy(), col)
        results['mean'] = abs(X_tr[col + '_mean'].std() - original_std)

        # ---------- MEDIAN ----------
        X_tr, X_te = median_imputation(X_train.copy(), X_test.copy(), col)
        results['median'] = abs(X_tr[col + '_median'].std() - original_std)

        # ---------- MODE ----------
        X_tr, X_te = mode_imputation(X_train.copy(), X_test.copy(), col)
        results['mode'] = abs(X_tr[col + '_mode'].std() - original_std)

        # ---------- CONSTANT ----------
        X_tr, X_te = constant_imputation(X_train.copy(), X_test.copy(), col)
        results['constant'] = abs(X_tr[col + '_const'].std() - original_std)

        # ---------- ARBITRARY ----------
        X_tr, X_te = arbitrary_imputation(X_train.copy(), X_test.copy(), col)
        results['arbitrary'] = abs(X_tr[col + '_arb'].std() - original_std)

        # ---------- END OF DISTRIBUTION ----------
        X_tr, X_te = eod_imputation(X_train.copy(), X_test.copy(), col)
        results['eod'] = abs(X_tr[col + '_eod'].std() - original_std)

        # ---------- RANDOM SAMPLE ----------
        X_tr, X_te = random_sample_imputation(X_train.copy(), X_test.copy(), col)
        results['random_sample'] = abs(X_tr[col + '_rsi'].std() - original_std)

        # ---------- FORWARD FILL ----------
        X_tr, X_te = ffill_imputation(X_train.copy(), X_test.copy(), col)
        results['ffill'] = abs(X_tr[col + '_ffill'].std() - original_std)

        # ---------- BACKWARD FILL ----------
        X_tr, X_te = bfill_imputation(X_train.copy(), X_test.copy(), col)
        results['bfill'] = abs(X_tr[col + '_bfill'].std() - original_std)

        # ---------- FIND BEST ----------
        best_method = min(results, key=results.get)
        logger.info(f'==============================================')
        logger.info(f'STD comparison for column: {col}')
        for k, v in results.items():
            logger.info(f'{k} difference: {v}')
        logger.info(f'BEST IMPUTATION METHOD for {col}: {best_method}')
        logger.info(f'==============================================')
        return best_method

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'error in line {ex_line.tb_lineno} due to {ex_msg}')

def apply_best_imputation(X_train, X_test, col):
    """
    Finds the best imputation method using STD
    and applies it on original column
    """
    try:
        best_method = find_best_imputation(X_train, X_test, col)

        logger.info(f'Applying best method "{best_method}" on column {col}')

        # ---------- APPLY BEST METHOD ----------
        if best_method == 'mean':
            value = X_train[col].mean()
            X_train[col] = X_train[col].fillna(value)
            X_test[col] = X_test[col].fillna(value)

        elif best_method == 'median':
            value = X_train[col].median()
            X_train[col] = X_train[col].fillna(value)
            X_test[col] = X_test[col].fillna(value)

        elif best_method == 'mode':
            value = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(value)
            X_test[col] = X_test[col].fillna(value)

        elif best_method == 'constant':
            X_train[col] = X_train[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)

        elif best_method == 'arbitrary':
            X_train[col] = X_train[col].fillna(-1)
            X_test[col] = X_test[col].fillna(-1)

        elif best_method == 'eod':
            eod = X_train[col].mean() + 3 * X_train[col].std()
            X_train[col] = X_train[col].fillna(eod)
            X_test[col] = X_test[col].fillna(eod)

        elif best_method == 'random_sample':
            sample_train = X_train[col].dropna().sample(
                X_train[col].isnull().sum(), replace=True, random_state=42
            )
            sample_train.index = X_train[X_train[col].isnull()].index
            X_train.loc[X_train[col].isnull(), col] = sample_train

            sample_test = X_train[col].dropna().sample(
                X_test[col].isnull().sum(), replace=True, random_state=42
            )
            sample_test.index = X_test[X_test[col].isnull()].index
            X_test.loc[X_test[col].isnull(), col] = sample_test

        elif best_method == 'ffill':
            X_train[col] = X_train[col].ffill()
            X_test[col] = X_test[col].ffill()

        elif best_method == 'bfill':
            X_train[col] = X_train[col].bfill()
            X_test[col] = X_test[col].bfill()

        logger.info(f'Null values after imputation (train): {X_train[col].isnull().sum()}')
        logger.info(f'Null values after imputation (test): {X_test[col].isnull().sum()}')

        return X_train, X_test

    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'error in line {ex_line.tb_lineno} due to {ex_msg}')





