Housing Price Prediction on Google Colab
Introduction

This project aims to build an end-to-end machine learning pipeline to predict housing prices in Germany. As a newly appointed data scientist, you are required to design and implement this pipeline and submit it to the team lead data scientist. The project will help the company gain a competitive advantage by accurately predicting property prices, leading to increased customer satisfaction and business growth.
How to Run the Project on Google Colab
Step-by-Step Instructions

    Open Google Colab:
        Go to Google Colab in your web browser.

    Create a New Notebook:
        Click on "File" -> "New Notebook".

    Import Libraries:
        Copy and paste the following code into a cell to import necessary libraries:

python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

    Load the Dataset:
        Copy and paste the following code to load the dataset:

python

df = pd.read_csv("https://raw.githubusercontent.com/SeverusSnapee/ML-PROJECT/main/Housing.csv")

    Preprocess the Data:
        Copy and paste the following code to preprocess the data:

python

# Convert categorical variables to numerical
df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})
df['basement'] = df['basement'].map({'yes': 1, 'no': 0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

# Remove outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# Split data into features and target
X = df.drop(columns=['price'])
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

    Train and Evaluate Models:
        Support Vector Regression (SVR) with Linear Kernel:

python

# SVR with Linear Kernel
svr_linear = SVR(kernel='linear')
svr_linear.fit(X_train, y_train)
y_pred_svr_linear = svr_linear.predict(X_test)
r2_svr_linear = r2_score(y_test, y_pred_svr_linear)
mae_svr_linear = mean_absolute_error(y_test, y_pred_svr_linear)
print(f"SVR Linear Kernel R² value: {r2_svr_linear}")
print(f"SVR Linear Kernel MAE value: {mae_svr_linear}")

    Gradient Boosting Regressor with Hyperparameter Tuning:

python

# Gradient Boosting Regressor with Hyperparameter Tuning
gbr = GradientBoostingRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.3, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
best_gbr = grid_search.best_estimator_

y_pred_gbr = best_gbr.predict(X_test)
r2_gbr = r2_score(y_test, y_pred_gbr)
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
print(f"Gradient Boosting Regressor R² value: {r2_gbr}")
print(f"Gradient Boosting Regressor MAE value: {mae_gbr}")
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

    Discuss Results:
        Copy and paste the following code to print a summary of the results:

python

print("Model Performance Summary:")
print(f"SVR Linear Kernel - R²: {r2_svr_linear}, MAE: {mae_svr_linear}")
print(f"Gradient Boosting Regressor - R²: {r2_gbr}, MAE: {mae_gbr}")

Conclusion

This project demonstrates an end-to-end machine learning pipeline for predicting housing prices using various models and hyperparameter tuning. The Gradient Boosting Regressor with hyperparameter tuning provided the best performance.

By following these steps, you can run the entire project on Google Colab and analyze the results. Feel free to experiment with different models and parameters to improve the performance further.

This README file provides detailed instructions on how to set up and run your project on Google Colab, making it easy for anyone to follow along.
