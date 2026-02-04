import sys
import numpy as np
import pandas as pd
from log_code import setup_logging
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

logger = setup_logging('feature_selection')


# ======================================================
# CONSTANT FEATURE SELECTION
# ======================================================
def constant_feature_selection(X_train_num, X_test_num):
    try:
        logger.info(f"Before Constant FS : {X_train_num.shape}")

        constant_reg = VarianceThreshold(0.0)
        constant_reg.fit(X_train_num)

        keep_cols = X_train_num.columns[constant_reg.get_support()]
        X_train_num = X_train_num[keep_cols]
        X_test_num = X_test_num[keep_cols]

        logger.info(f"After Constant FS : {X_train_num.shape}")
        logger.info(f"Remaining columns : {list(keep_cols)}")
        logger.info("=" * 60)

        return X_train_num, X_test_num

    except Exception:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Constant FS error line {er_line.tb_lineno} due to {er_msg}")


# ======================================================
# QUASI-CONSTANT FEATURE SELECTION
# ======================================================
def quasi_constant_feature_selection(X_train_num, X_test_num, threshold=0.1):
    try:
        logger.info(f"Before Quasi-Constant FS : {X_train_num.shape}")

        quasi_constant_reg = VarianceThreshold(threshold)
        quasi_constant_reg.fit(X_train_num)

        keep_cols = X_train_num.columns[quasi_constant_reg.get_support()]
        X_train_num = X_train_num[keep_cols]
        X_test_num = X_test_num[keep_cols]

        logger.info(f"After Quasi-Constant FS : {X_train_num.shape}")
        logger.info(f"Remaining columns : {list(keep_cols)}")
        logger.info("=" * 60)

        return X_train_num, X_test_num

    except Exception:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"Quasi-Constant FS error line {er_line.tb_lineno} due to {er_msg}")


# ======================================================
# HYPOTHESIS TESTING (PEARSON CORRELATION)
# ======================================================
from scipy.stats import pearsonr
import sys
'''
def hypothesis_testing_fs(X_train_num, X_test_num, y_train, p_threshold=0.05):
    try:
        logger.info("Starting Hypothesis Testing (Pearson)")

        # simple target encoding (for churn: Yes/No)
        y_encoded = y_train.map({'Yes': 1, 'No': 0})

        selected_cols = []
        dropped_cols = []

        for col in X_train_num.columns:

            # ✅ SIMPLE CHECK
            if X_train_num[col].nunique() <= 1:
                logger.info(f"Skipping {col} (constant column)")
                dropped_cols.append(col)
                continue

            # Pearson correlation
            _, p_val = pearsonr(X_train_num[col], y_encoded)

            if p_val <= p_threshold:
                selected_cols.append(col)
            else:
                dropped_cols.append(col)

        # keep only selected columns
        X_train_num = X_train_num[selected_cols]
        X_test_num = X_test_num[selected_cols]

        logger.info(f"Selected features : {selected_cols}")
        logger.info(f"Dropped features  : {dropped_cols}")
        logger.info(f"After Hypothesis FS : {X_train_num.shape}")
        logger.info("=" * 50)

        return X_train_num, X_test_num

    except Exception:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(
            f"Hypothesis FS error line {er_line.tb_lineno} due to {er_msg}"
        )
        return X_train_num, X_test_num
'''
# ======================================================
# MASTER FUNCTION (OPTIONAL)
# ======================================================
def all_selections(X_train_num, X_test_num, y_train):
    try:
        X_train_num, X_test_num = constant_feature_selection(
            X_train_num, X_test_num
        )

        X_train_num, X_test_num = quasi_constant_feature_selection(
            X_train_num, X_test_num
        )

       # X_train_num, X_test_num = hypothesis_testing_fs(X_train_num, X_test_num, y_train)

        logger.info("Feature selection completed successfully")
        return X_train_num, X_test_num

    except Exception:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.info(f"All FS error line {er_line.tb_lineno} due to {er_msg}")
