import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('all_models')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
import pickle

def knn(X_train,y_train,X_test,y_test):
    try:
      global knn_reg
      knn_reg = KNeighborsClassifier(n_neighbors=5)
      knn_reg.fit(X_train,y_train)
      logger.info(f'KNN Test Accuracy : {accuracy_score(y_test,knn_reg.predict(X_test))}')
      logger.info(f'KNN confusion matrix : {confusion_matrix(y_test, knn_reg.predict(X_test))}')
      logger.info(f'KNN classification report : {classification_report(y_test, knn_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def nb(X_train,y_train,X_test,y_test):
    try:
      global naive_reg
      naive_reg = GaussianNB()
      naive_reg.fit(X_train,y_train)
      logger.info(f'Naive Bayes Test Accuracy : {accuracy_score(y_test,naive_reg.predict(X_test))}')
      logger.info(f'Naive Bayes confusion matrix : {confusion_matrix(y_test, naive_reg.predict(X_test))}')
      logger.info(f'Naive Bayes classification report : {classification_report(y_test, naive_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def lr(X_train,y_train,X_test,y_test):
    try:
      global lr_reg
      lr_reg = LogisticRegression()
      lr_reg.fit(X_train,y_train)
      logger.info(f'LogisticRegression Test Accuracy : {accuracy_score(y_test,lr_reg.predict(X_test))}')
      logger.info(f'LogisticRegression confusion matrix : {confusion_matrix(y_test, lr_reg.predict(X_test))}')
      logger.info(f'LogisticRegression Test Accuracy : {classification_report(y_test, lr_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def dt(X_train,y_train,X_test,y_test):
    try:
      global dt_reg
      dt_reg = DecisionTreeClassifier(criterion='entropy')
      dt_reg.fit(X_train,y_train)
      logger.info(f'DecisionTreeClassifier Test Accuracy : {accuracy_score(y_test,dt_reg.predict(X_test))}')
      logger.info(f'DecisionTreeClassifier confusion matrix : {confusion_matrix(y_test, dt_reg.predict(X_test))}')
      logger.info(f'DecisionTreeClassifier classification report : {classification_report(y_test,dt_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def rf(X_train,y_train,X_test,y_test):
    try:
      global rf_reg
      rf_reg = RandomForestClassifier(n_estimators=5,criterion='entropy')
      rf_reg.fit(X_train,y_train)
      logger.info(f'RandomForestClassifier Test Accuracy : {accuracy_score(y_test,rf_reg.predict(X_test))}')
      logger.info(f'RandomForestClassifier confusion matrix : {confusion_matrix(y_test, rf_reg.predict(X_test))}')
      logger.info(f'RandomForestClassifier classification report : {classification_report(y_test, rf_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def common(X_train,y_train,X_test,y_test):
    try:
      logger.info('=========KNN===========')
      knn(X_train,y_train,X_test,y_test)
      logger.info('=========NB===========')
      nb(X_train,y_train,X_test,y_test)
      logger.info('=========LR===========')
      lr(X_train,y_train,X_test,y_test)
      logger.info('=========DT===========')
      dt(X_train,y_train,X_test,y_test)
      logger.info('=========RF===========')
      rf(X_train,y_train,X_test,y_test)
      knn_predictions = knn_reg.predict_proba(X_test)[:, 1]
      naive_predictions = naive_reg.predict_proba(X_test)[:, 1]
      lr_predictions = lr_reg.predict_proba(X_test)[:, 1]
      with open('churn.pkl','wb') as f:
          pickle.dump(lr_reg,f)
      dt_predictions = dt_reg.predict_proba(X_test)[:, 1]
      rf_predictions = rf_reg.predict_proba(X_test)[:, 1]

      knn_fpr, knn_tpr, knn_thre = roc_curve(y_test, knn_predictions)
      nb_fpr, nb_tpr, nb_thre = roc_curve(y_test, naive_predictions)
      lr_fpr, lr_tpr, lr_thre = roc_curve(y_test, lr_predictions)
      dt_fpr, dt_tpr, dt_thre = roc_curve(y_test, dt_predictions)
      rf_fpr, rf_tpr, rf_thre = roc_curve(y_test, rf_predictions)
      plt.figure(figsize=(5, 3))
      plt.plot([0, 1], [0, 1], "k--")

      plt.plot(knn_fpr, knn_tpr, label="KNN")
      plt.plot(nb_fpr, nb_tpr, label="NB")
      plt.plot(lr_fpr, lr_tpr, label="LR")
      plt.plot(dt_fpr, dt_tpr, label="DT")
      plt.plot(rf_fpr, rf_tpr, label="RF")

      plt.xlabel("FPR")
      plt.ylabel("TPR")
      plt.title("ROC Curve - ALL Models")
      plt.legend(loc=0)
      plt.show()


    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')