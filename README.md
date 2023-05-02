# Industrial-Copper-Modeling
**Problem Statement:**
    The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data. 
    Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.

**Approach:** 
Data Understanding: Identify the types of variables (continuous, categorical) and their distributions. Some rubbish values are present in ‘Material_Reference’ which starts with ‘00000’ value which should be converted into null.
Data Preprocessing: 
Handle missing values with mean/median/mode.
Treat Outliers using IQR or Isolation Forest from sklearn library.
Identify Skewness in the dataset and treat skewness with appropriate data transformations, such as log transformation(which is best suited to transform target variable-train, predict and then reverse transform it back to original scale eg:dollars), to handle high skewness in continuous variables.
Encode categorical variables using one-hot encoding label encoding technique.
EDA: Try visualizing outliers and skewness(before and after treating skewness) using Seaborn’s boxplot, distplot.
Feature Engineering: Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data. And drop highly correlated columns using SNS HEATMAP.
Model Building and Evaluation:
Split the dataset into training and testing/validation sets. 
Train and evaluate different classification models using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve. 
Optimize model hyperparameters using grid search technique to find the best-performing model.
Interpret the model results and assess its performance based on the defined problem statement.
Same steps for Regression modelling.Decision Tree Regressor is used.
Model GUI: Using streamlit module, created interactive page with
   (1) task input( Regression or Classification) and 
   (2) In input field where you can enter each column value except ‘Selling_Price’ for regression model and  except ‘Status’ for classification model. 
   
**Libraries used:**
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelBinarizer

import streamlit as st

import re

**Installation:**
  To use sklearn package we have to install pip install -U scikit-learn
