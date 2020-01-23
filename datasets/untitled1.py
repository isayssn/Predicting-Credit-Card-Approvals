# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:40:50 2020

@author: ISA YASASİN
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:59:36 2020

@author: ISA YASASİN
"""

import pandas as pd
import numpy as np

header=["Gender", "Age", "Debt", "Married", 
        "BankCustomer", "EducationLevel", "Ethnicity", 
        "YearsEmployed", "PriorDefault", "Employed", 
        "CreditScore", "DriversLicense", "Citizen",
        "ZipCode", "Income","ApprovalStatus"]
df = pd.read_csv("cc_approvals.data",names=header)
df.info()
df.describe()

#%% ? => NaN
data = df.replace("?",np.NaN)
data.info()

#%% Filling NaN in int,float with mean 
pd.isna(data).sum()
data.fillna(data.mean(),inplace=True)
data.isnull().values.sum()
pd.isna(data).sum()

#%% Filling NaN in object type with most frequent datas
for col in data.columns:   
   if data[col].dtypes == "object":
      data = data.fillna(data[col].value_counts().index[0])

data.isnull().values.sum()

#%% Dropping "b" 
data.drop(data[data["Age"] == "b"].index, inplace=True)
data.drop(data[data["ZipCode"] == "b"].index, inplace=True)
data["Age"] = data["Age"].astype(float)
data["ZipCode"] = data["ZipCode"].astype(float)
data.info()
#%% object => int,float  (Age and ZipCode object They need to be protected from encoding)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data["Age"] = data.Age.astype(float)
data["ZipCode"] = data.ZipCode.astype(float)
for col in data.columns:
   if data[col].dtypes == "object":
      data[col] = le.fit_transform(data[col])
data.info()  
#%% train test split
from sklearn.model_selection import train_test_split
data = data.drop(["CreditScore","Citizen"],axis=1)  

x = data.iloc[:,0:13].values
y = data.iloc[:,13].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 42)
#%% Standardization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_x_train = scaler.fit_transform(x_train)
rescaled_x_test = scaler.fit_transform(x_test)

#%% Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')
lr.fit(rescaled_x_train,y_train)

#%% Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = lr.predict(rescaled_x_test)
c = confusion_matrix(y_test, y_pred)
print("Accuracy of logistic regression classifier:")
print(c)

#%% Grid Search
from sklearn.model_selection import GridSearchCV
tol = [0.01,0.001,0.0001]
max_iter = [100,150,200]
param_grid = dict(tol=tol,max_iter=max_iter)
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)
rescaled_x = scaler.fit_transform(x)
grid_model_result = grid.fit(rescaled_x, y)
best_score = grid_model_result.best_score_
best_parameters = grid_model_result.best_params_
print("Best Score:",best_score)
print(("Best Parameters:"), best_parameters)
























