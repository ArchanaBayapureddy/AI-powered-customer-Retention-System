import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from log_code import setup_logging
logger=setup_logging('main')
from missing_values import apply_best_imputation
from variable_transformation import vt
from outlier_handling import percentile_capping
from outlier_handling import (iqr_capping,zscore_outlier,percentile_capping,trimming)
from feature_selection import all_selections
from cat_to_num import changing_data_to_num
from balancing import scale_and_balance
from all_models import common
from tuning import tune_logistic



class CHURN:
    def __init__(self,path):
        try:
            self.df=pd.read_csv(path)
            sim_providers = ['Airtel', 'BSNL', 'Jio', 'Vodafone']
            self.df['Sim_Providers'] = np.random.choice(sim_providers, size=len(self.df))

            logger.info(f'{self.df.head()}')
            #logger.info(f'{self.df.info()}')
            logger.info(f'{self.df.shape}')
            logger.info(f'{self.df.dtypes}')


            logger.info(f'{self.df.isnull().sum()}')

            self.df=self.df.drop(['customerID'],axis=1)

            self.df['TotalCharges']=pd.to_numeric(self.df['TotalCharges'],errors='coerce')
            logger.info(f'==========================================================')

            logger.info(f'{self.df.sample(10)}')
            #logger.info(f'{self.df.info()}')
            logger.info(f'{self.df.isnull().sum()}')
            self.X=self.df.drop(['Churn'],axis=1)
            self.y=self.df.iloc[:,-2]
            logger.info(f'columnn names in dataset :{self.X.columns}')
            logger.info(f'shape of x:{self.X.shape}')
            logger.info(f'shape of y:{self.y}')
            logger.info(f'{self.y.shape}')
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f'X trainn shape :{self.X_train.shape}')
            logger.info(f'X test  shape :{self.X_test.shape}')

            logger.info(f'y train shape:{self.y_train.shape}')
            logger.info(f'y test shape {self.y_test.shape}')
            logger.info(f'{self.df.head()}')
            self.y_train = self.y_train.map({'No': 0, 'Yes': 1})
            self.y_test = self.y_test.map({'No': 0, 'Yes': 1})
            logger.info(f'null values in y_train:{self.y_train.isnull().sum()}')
        except Exception as e:
            ex_type,ex_msg,ex_line=sys.exc_info()
            logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')


    def null_values (self):
        try:
            logger.info(f'Before imputation:\n{self.X_train.isnull().sum()}')

            for col in self.X_train.columns:
                if self.X_train[col].isnull().sum() > 0:
                    self.X_train, self.X_test = apply_best_imputation(
                        self.X_train, self.X_test, col
                    )

            logger.info(f'After imputation:\n{self.X_train.isnull().sum()}')
            logger.info(f'After imputation:\n{self.X_test.isnull().sum()}')
        except Exception as e:
            ex_type,ex_msg,ex_line=sys.exc_info()
            logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')
    def variable_transformations(self):

        # ================= SPLIT NUMERIC & CATEGORICAL =================
        self.X_train_num = self.X_train.select_dtypes(include=['int64', 'float64'])
        self.X_test_num = self.X_test.select_dtypes(include=['int64', 'float64'])

        self.X_train_cat = self.X_train.select_dtypes(exclude=['int64', 'float64'])
        self.X_test_cat = self.X_test.select_dtypes(exclude=['int64', 'float64'])

        logger.info(f'X_train_num shape before transformation : {self.X_train_num.shape}')
        logger.info(f'X_train_cat shapebefore transformation : {self.X_train_cat.shape}')
        logger.info(f'X_train_cat columnsbefore transformation : {self.X_train_cat.columns}')

        self.X_train_num, self.X_test_num = vt(self.X_train_num, self.X_test_num)

        logger.info(f'X_train_num shape after transformation : {self.X_train_num.shape}')
        logger.info(f'X_train_cat shape after transformation : {self.X_train_cat.shape}')
        logger.info(f'x_train_num columns : {self.X_train_num.columns}')

        logger.info(f' X_train_cat  {self.X_train_cat.isnull().sum()}')
        logger.info(f' X_test_cat  {self.X_test_cat.isnull().sum()}')
        logger.info(f' X_train_num  {self.X_train_num.isnull().sum()}')
        logger.info(f' X_test_num  {self.X_test_num.isnull().sum()}')

    def outliers_handle(self):
        try:
            # STEP 1: Extract numerical data ONCE
            X_train_num = self.X_train.select_dtypes(include=['int64', 'float64'])
            X_test_num = self.X_test.select_dtypes(include=['int64', 'float64'])

            logger.info("Starting outlier plotting for all methods")

            # STEP 2: IQR (plots BEFORE & AFTER)
            iqr_capping(X_train_num.copy(),X_test_num.copy())

            # STEP 3: Z-SCORE (plots BEFORE & AFTER)
            zscore_outlier( X_train_num.copy(),X_test_num.copy())

            # STEP 4: PERCENTILE (plots BEFORE & AFTER)
            percentile_capping(X_train_num.copy(),X_test_num.copy())

            # STEP 5: TRIMMING (plots BEFORE & AFTER)
            trimming(X_train_num.copy(),X_test_num.copy())
            # =================================================
            # APPLY ONE BEST METHOD TO ACTUAL DATA
            # =================================================
            logger.info("Applying IQR capping to actual data")

            self.X_train_num, self.X_test_num = iqr_capping(
                X_train_num.copy(),
                X_test_num.copy()
            )

            logger.info("Outlier handling applied to training data")
            logger.info(f'X_train columns shape: {self.X_train_num.columns}')
            logger.info(f'X_test columns : {self.X_test_num.columns}')
            logger.info("All outlier plots generated successfully")

        except Exception as e:
            ex_type, ex_msg, ex_line = sys.exc_info()
            logger.info(
                f"Error in outliers_handle line {ex_line.tb_lineno} due to {ex_msg}"
            )

    def feature_selection(self):
        try:
            self.X_train_num, self.X_test_num = all_selections(self.X_train_num, self.X_test_num,self.y_train)

            logger.info("Feature selection completed")
            logger.info(f"Final selected features : {self.X_train_num.columns}")
            logger.info(f'datatype of numerical columns : {self.X_train_num.dtypes}')

        except Exception:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"FS error line {er_line.tb_lineno} due to {er_msg}")

    def cat_to_num(self):
        try:
            logger.info("Before Converting into Numeircal")
            logger.info(f"{self.X_train_cat.columns}")
            logger.info(f"{self.X_train_cat.isnull().sum()}")
            logger.info("Before Converting into Numeircal")
            logger.info(f"{self.X_test_cat.columns}")
            logger.info(f"{self.X_test_cat.isnull().sum()}")

            self.X_train_cat, self.X_test_cat = changing_data_to_num(self.X_train_cat, self.X_test_cat)

            logger.info("After Converting into Numeircal")
            logger.info(f"{self.X_train_cat.columns}")
            logger.info(f"{self.X_train_cat.isnull().sum()}")
            logger.info("After Converting into Numeircal")
            logger.info(f"{self.X_test_cat.columns}")
            logger.info(f"{self.X_test_cat.isnull().sum()}")
            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)

            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

            logger.info(f"Final Training data")
            logger.info(f"{self.training_data.columns}")
            logger.info(f"{self.training_data.sample(10)}")
            logger.info(f"{self.training_data.isnull().sum()}")

            logger.info(f"Final Testing data")
            logger.info(f"{self.testing_data.columns}")
            logger.info(f"{self.testing_data.sample(10)}")
            logger.info(f"{self.testing_data.isnull().sum()}")
            logger.info(f"{self.training_data.shape}")
            logger.info(f"{self.testing_data.shape}")
            logger.info(f"{self.y_train.shape}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

    def data_balancing(self):
        try:
            logger.info("\n" + self.y_train.head(5).to_string())

            self.X_train , self.X_test , self.y_train = scale_and_balance(
                self.training_data,
                self.testing_data,
               self.y_train,
            scaler_path="scaler.pkl" )
            logger.info("Data balancing and scaling completed successfully")
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")

    def train_model(self):
        try:
            common(self.X_train, self.y_train, self.X_test, self.y_test)
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")
    def tuning(self):
        try:
            best_model, best_params, best_score, test_roc_auc = tune_logistic(self.X_train, self.y_train)
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no : {er_line.tb_lineno} : due to {er_msg}")



if __name__ == '__main__':
    try:
        path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
        obj = CHURN(path)
        obj.null_values()
        obj.variable_transformations()
        obj.outliers_handle()
        obj.feature_selection()
        obj.cat_to_num()
        obj.data_balancing()
        obj.train_model()
        obj.tuning()
    except Exception as e:
        ex_type, ex_msg, ex_line = sys.exc_info()
        logger.info(f'the error is from {ex_line.tb_lineno} due to {ex_msg}')
