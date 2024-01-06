# Import the package to use
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# data load
train = pd.read_csv("/Users/a81060586/modelAsPackage/train.csv").drop('id', axis=1)
test = pd.read_csv('/Users/a81060586/modelAsPackage/test.csv')


# Unified data format
y_train = train['target']
X_train = train.drop('target', axis=1)
X_test = test.drop('id', axis = 1)

print(X_train.shape)

print(X_test.shape)

# ideally should be checking the distribution before selecting the scaler
# Standardize data
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)



# Using grid search to find the best parameters of xgboost
xgb_model = xgb.XGBClassifier(random_state=42)
param_grid = {'objective':['binary:logistic'],
              'learning_rate': [0.001,0.05,0.1, 10], 
              'max_depth': [2,3,4,5,6],
              'min_child_weight': [11],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [1000]}

grid = GridSearchCV(estimator = xgb_model, cv=5, param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1, refit=True)
grid.fit(X_train,y_train)

print("Best Score:" + str(grid.best_score_))
print("Best Parameters: " + str(grid.best_params_))

best_parameters = grid.best_params_


# Recursive feature elimination for xgboost models using RFE functions (select the top 100)
xgb_model = xgb.XGBClassifier(**best_parameters)
xgb_model.fit(X_train,y_train)

selector = RFE(estimator=xgb_model, n_features_to_select=100, step=1)
selector.fit(X_train,y_train)

xgb_preds = selector.predict_proba(X_test)[:,1]

train_predict = selector.predict(X_train)
print(roc_auc_score(y_train, train_predict))



# Using Grid Search to Find the Best Parameters of a Logical Regression Model
lr = LogisticRegression(random_state=42)
param_grid = {'class_weight' : ['balanced', None], 
              'penalty' : ['l2','l1'], 
              'C' : [0.001, 0.01, 0.1, 1, 10, 100],
              'solver': ['saga']}

grid = GridSearchCV(estimator = lr, cv=5, param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1)
grid.fit(X_train,y_train)

print("Best Score:" + str(grid.best_score_))
print("Best Parameters: " + str(grid.best_params_))

best_parameters = grid.best_params_


# Recursive feature elimination using RFE functions for logical regression models (select the first 150)
lr = LogisticRegression(**best_parameters)
lr.fit(X_train,y_train)

selector = RFE(estimator=lr, n_features_to_select=150, step=1)
selector.fit(X_train,y_train)

lr_preds = lr.predict_proba(X_test)[:,1]

train_predict = lr.predict(X_train)
print(roc_auc_score(y_train, train_predict))

# Model fusion, with higher accuracy models given higher weights, and finally prediction
final_preds = (lr_preds * 0.8 + xgb_preds * 0.2)
submission = pd.read_csv('/Users/a81060586/modelAsPackage/sample_submission.csv')
submission['target'] = final_preds
submission.to_csv('submission.csv', index=False)
print(submission.head(20))


