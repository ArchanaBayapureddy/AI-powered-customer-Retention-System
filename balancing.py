import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import logging
import seaborn as sns
from log_code import setup_logging
logger = setup_logging('balancing')
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle

def scale_and_balance(train_data, test_data, y_train, scaler_path="scaler.pkl"):
    try:

        logger.info("===== Starting AUTO Balancing + Scaling =====")

        # Ensure y_train is Series
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        logger.info("\n" + y_train.head(5).to_string())

        logger.info(
            "Target BEFORE SMOTE:\n" +
            y_train.value_counts().to_string()
        )

        # -------------------
        # 1. SMOTE (TRAIN ONLY)
        # -------------------
        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(train_data, y_train)

        logger.info(
            "Target AFTER SMOTE:\n" +
            pd.Series(y_bal).value_counts().to_string()
        )
        logger.info(f"Shape after SMOTE: {X_bal.shape}")



        # -------------------
        # 2. Scaling
        # -------------------
        logger.info("Starting feature scaling using StandardScaler")

        scaler = StandardScaler()

        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_bal),
            columns=X_bal.columns
        )

        X_test_scaled = pd.DataFrame(
            scaler.transform(test_data),
            columns=test_data.columns,
            index=test_data.index
        )

        # -------------------
        # 3. Save Scaler
        # -------------------
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f'Scaled cols : {X_train_scaled.head(5)}')
        logger.info("Scaler saved successfully")

        logger.info("===== Balancing + Scaling Completed =====")

        return X_train_scaled, X_test_scaled, y_bal

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.error(
            f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}"
        )
        raise e


