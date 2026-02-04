# AI-powered-customer-Retention-System
## 📌 Project Overview

* This project focuses on predicting **customer churn** using Machine Learning techniques.
*Customer churn refers to customers who are likely to stop using a company’s service.
*The main objective of this project is to identify such customers in advance so that businesses can take necessary actions to improve customer retention.
*The project implements a **complete end-to-end machine learning pipeline**, starting from data preprocessing to model deployment. It includes handling missing values, feature transformation, feature selection, *encoding categorical variables, data balancing, feature scaling, model training, and evaluation.
*After comparing multiple machine learning models, **Logistic Regression** is selected as the final model due to its strong performance on binary classification problems and good ROC-AUC score.
*The trained model and preprocessing objects are saved and can be directly used for deployment in real-world applications.

## 📊 Dataset Description
The dataset used in this project is the **Telco Customer Churn dataset**, which contains customer-level information for predicting churn.
**📁 Dataset Size**
* **Number of Rows:** 7,043
* **Number of Columns:** 21
Each row represents **one customer**, and each column represents a **customer attribute**.

### 🎯 Target Variable
* **`Churn`**
  * `Yes` → Customer has churned
  * `No` → Customer has not churned
   
###  Feature Variables
**Customer Demographics**

* `customerID`
* `gender`
* `SeniorCitizen`
* `Partner`
* `Dependents`

**Service Information**

* `PhoneService`
* `MultipleLines`
* `InternetService`
* `OnlineSecurity`
* `OnlineBackup`
* `DeviceProtection`
* `TechSupport`
* `StreamingTV`
* `StreamingMovies`

**Account and Billing Information**

* `tenure`
* `Contract`
* `PaperlessBilling`
* `PaymentMethod`
* `MonthlyCharges`
* `TotalCharges`
### Dataset Characteristics

* Contains both numerical and categorical features.
* Includes missing values (handled during preprocessing)
* Target variable is imbalanced
* Suitable for binary classification problems

### 🛠️ Technologies Used

* **Python** – Core programming language used to build the project
* **Pandas** – Used for data loading, cleaning, and manipulation
* **NumPy** – Used for numerical computations and array operations
* **Matplotlib** – Used for data visualization and plotting graphs
* **Seaborn** – Used for statistical data visualization
* **Scikit-learn** – Used for data preprocessing, feature selection, model training, and evaluation
* **Pickle** – Used to save trained models and preprocessing objects
* **Logging** – Used to track execution flow and handle errors
   
## 🧩 Handling Missing Values
Missing values in the dataset are handled using Mode Imputation.
In this method, missing values are replaced with the most frequently occurring value in each feature.
This method is finalized because when comparing the its standard deviation to original standard deviation its score is low .

## 🔄 Variable Transformation
Variable transformation is applied to numerical features to improve data distribution and enhance model performance.
In this project, transformation techniques are used to reduce skewness and make the data more suitable for machine learning algorithms.

**Selected Transformations**:
SeniorCitizen → Log Transformation
tenure → No transformation (original values retained)
MonthlyCharges → No transformation (original values retained)
TotalCharges → Yeo-Johnson Transformation

## 🚨 Outlier Handling
Outliers are handled using the **Interquartile Range (IQR) Capping** method.
In this technique, lower and upper bounds are calculated using Q1 and Q3. Values falling outside these bounds are capped to the nearest limit instead of being removed.
**Why IQR Capping ??**
* Preserves all data points
* Reduces impact of extreme values
* Improves model stability

## 🎯 Feature Selection
Feature selection is applied to **numerical features only** to remove uninformative variables and improve model performance.
In this project, **variance-based feature selection techniques** are used.
**Techniques Used:**

**1. Constant Feature Removal**
* Features with zero variance are removed
* Such features contain the same value for all records and provide no predictive power
  
**2. Quasi-Constant Feature Removal**
* Features with very low variance are removed using a threshold
* These features carry very little information and may introduce noise
Both techniques are implemented using VarianceThreshold and are applied only on the training data. The same selected features are then applied to the test data to avoid data leakage.

## 🔠 Categorical to Numerical Conversion
Categorical variables are converted into numerical format to make the data suitable for machine learning models.
In this project:
* **One-Hot Encoding** is used for nominal categorical features
* **Ordinal Encoding** is used for ordinal categorical features
Encoding is applied by fitting on the training data and transforming both training and test datasets to avoid data leakage.
This step ensures the dataset is fully numerical and ready for further processing and model training.

## ⚖️ Data Balancing
The dataset contains an imbalanced target variable, so **SMOTE (Synthetic Minority Over-sampling Technique)** is used to balance the data.
SMOTE is applied **only on the training dataset** to generate synthetic samples for the minority class. This ensures that the model learns equally from both classes and avoids bias toward the majority class.
Balancing is performed **before feature scaling** to ensure newly generated samples are also scaled properly.

## 📏 Feature Scaling
Feature scaling is applied using **StandardScaler** to standardize numerical features.
StandardScaler transforms the data so that features have:
* Mean = 0
* Standard Deviation = 1
The scaler is **fitted on the balanced training data** and then applied to the test data to prevent data leakage.
The trained scaler is saved as a `.pkl` file for use during deployment.


## 🤖 Model Training
The churn prediction model is trained using **Logistic Regression**, which is selected as the final model for this project.
Logistic Regression is well suited for **binary classification problems** like customer churn. The model is trained on the fully preprocessed and balanced dataset and evaluated on the test data.

**Why Logistic Regression**
* Performs well for binary classification
* Produces stable and consistent results
* Easy to interpret
* Shows good ROC-AUC performance
The trained Logistic Regression model is saved as a `.pkl` file and used for deployment.

## ⚙️ Hyperparameter Tuning
Hyperparameter tuning is performed to optimize the performance of the **Logistic Regression** model.
In this project, **GridSearchCV** is used to find the best combination of hyperparameters. The model is evaluated using **ROC-AUC score** with **5-fold cross-validation**.

**Parameters Tuned**
* Regularization strength (`C`)
* Penalty type (`l1`, `l2`, `elasticnet`, `none`)
* Solver (`liblinear`, `saga`)
* Maximum number of iterations (`max_iter`)

## 🚀 Deployment
The trained churn prediction model is deployed using Render.com.
The saved Logistic Regression model (.pkl), along with the scaler and encoders, is loaded in the deployment environment to make predictions on new customer data.

**Deployemet link** :












