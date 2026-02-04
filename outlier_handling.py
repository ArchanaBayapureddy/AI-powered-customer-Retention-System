import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from log_code import setup_logging

logger = setup_logging('outlier_handling')

# ======================================================
# COMMON PATH
# ======================================================
PLOT_PATH = "plots"
os.makedirs(PLOT_PATH, exist_ok=True)


# ======================================================
# COMMON BOXPLOT FUNCTION
# ======================================================
def save_boxplots(df, method, stage):
    for col in df.columns:
        plt.figure(figsize=(5, 4))
        sns.boxplot(x=df[col])
        plt.title(f"{method} {stage} - {col}")
        plt.savefig(f"{PLOT_PATH}/{method}_{stage}_{col}.png")
        plt.close()
def iqr_capping(X_train_num, X_test_num):
    save_boxplots(X_train_num, "IQR", "BEFORE")

    for col in list(X_train_num.columns):
        Q1 = X_train_num[col].quantile(0.25)
        Q3 = X_train_num[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # create NEW column
        X_train_num[col + "_iqr"] = np.where(
            X_train_num[col] < lower, lower,
            np.where(X_train_num[col] > upper, upper, X_train_num[col])
        )

        X_test_num[col + "_iqr"] = np.where(
            X_test_num[col] < lower, lower,
            np.where(X_test_num[col] > upper, upper, X_test_num[col])
        )

        # drop ORIGINAL
        X_train_num.drop(col, axis=1, inplace=True)
        X_test_num.drop(col, axis=1, inplace=True)

    save_boxplots(X_train_num, "IQR", "AFTER")
    return X_train_num, X_test_num
def zscore_outlier(X_train_num, X_test_num, threshold=3):
    save_boxplots(X_train_num, "ZSCORE", "BEFORE")

    for col in list(X_train_num.columns):
        mean = X_train_num[col].mean()
        std = X_train_num[col].std()

        z_tr = abs((X_train_num[col] - mean) / std)
        z_te = abs((X_test_num[col] - mean) / std)

        X_train_num[col + "_z"] = np.where(z_tr > threshold, mean, X_train_num[col])
        X_test_num[col + "_z"] = np.where(z_te > threshold, mean, X_test_num[col])

        X_train_num.drop(col, axis=1, inplace=True)
        X_test_num.drop(col, axis=1, inplace=True)

    save_boxplots(X_train_num, "ZSCORE", "AFTER")
    return X_train_num, X_test_num
def percentile_capping(X_train_num, X_test_num, lower_pct=0.05, upper_pct=0.95):
    save_boxplots(X_train_num, "PERCENTILE", "BEFORE")

    for col in list(X_train_num.columns):
        lower = X_train_num[col].quantile(lower_pct)
        upper = X_train_num[col].quantile(upper_pct)

        X_train_num[col + "_pct"] = np.where(
            X_train_num[col] < lower, lower,
            np.where(X_train_num[col] > upper, upper, X_train_num[col])
        )

        X_test_num[col + "_pct"] = np.where(
            X_test_num[col] < lower, lower,
            np.where(X_test_num[col] > upper, upper, X_test_num[col])
        )

        X_train_num.drop(col, axis=1, inplace=True)
        X_test_num.drop(col, axis=1, inplace=True)

    save_boxplots(X_train_num, "PERCENTILE", "AFTER")
    return X_train_num, X_test_num
def trimming(X_train_num, X_test_num):
    save_boxplots(X_train_num, "TRIMMING", "BEFORE")

    for col in list(X_train_num.columns):
        Q1 = X_train_num[col].quantile(0.25)
        Q3 = X_train_num[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        X_train_num[col + "_trim"] = np.where(
            X_train_num[col] < lower, lower,
            np.where(X_train_num[col] > upper, upper, X_train_num[col])
        )

        X_test_num[col + "_trim"] = np.where(
            X_test_num[col] < lower, lower,
            np.where(X_test_num[col] > upper, upper, X_test_num[col])
        )

        X_train_num.drop(col, axis=1, inplace=True)
        X_test_num.drop(col, axis=1, inplace=True)

    save_boxplots(X_train_num, "TRIMMING", "AFTER")
    return X_train_num, X_test_num
