#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:46:13 2020

@author: xuebinwang
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("housing.csv")
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Draw scatter plot
cols = ['LSTAT', 'INDUS','NOX', 'RM', 'MEDV']
sns.pairplot(df[cols])
plt.show()


# Draw heatmap
cm = np.corrcoef(df.values.T)
hm = sns.heatmap(cm, cbar = True, annot = True, yticklabels=  df.columns, xticklabels=df.columns)
plt.show()

#Linear model
X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
sc.fit(X_train)
sc.fit(X_test)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
lr = LinearRegression()
lr.fit(X_train_std, y_train)
y_train_predlr = lr.predict(X_train_std)
y_test_predlr = lr.predict(X_test_std)
print(lr.coef_)
print(lr.intercept_)
print()


plt.scatter(y_train_predlr,  y_train_predlr - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_predlr,  y_test_predlr - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values using linear regression')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()


print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_predlr),
        mean_squared_error(y_test, y_test_predlr)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_predlr),
        r2_score(y_test, y_test_predlr)))


# Lasso


lassocv = LassoCV(alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],cv= 10)

lassocv.fit(X_train_std, y_train)
print("The best alpha is ", lassocv.alpha_)

lasso = Lasso(alpha = lassocv.alpha_)
lasso.fit(X_train_std, y_train)

y_test_predl = lasso.predict(X_test_std)
y_train_predl = lasso.predict(X_train_std)

print(lasso.coef_, lasso.intercept_)



print('MSE train: %.3f, test3: %.3f' % (
        mean_squared_error(y_train, y_train_predl),
        mean_squared_error(y_test, y_test_predl)))


print('R^2 train: %.3f, test3: %.3f' % (
        r2_score(y_train, y_train_predl),
        r2_score(y_test, y_test_predl)))

plt.scatter(y_train_predl,  y_train_predl - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_predl,  y_test_predl - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values using Lasso')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()



#Ridge



ridgecv = RidgeCV(alphas = [1,2,3,4,5,6,7,8,9,10],cv = 10)

ridgecv.fit(X_train_std, y_train)
print("The best alpha is ", ridgecv.alpha_)

ridge = Ridge(alpha = ridgecv.alpha_)
ridge.fit(X_train_std, y_train)

y_test_predr = ridge.predict(X_test_std)
y_train_predr = ridge.predict(X_train_std)

print(ridge.coef_, ridge.intercept_)



print('MSE train: %.3f, test3: %.3f' % (
        mean_squared_error(y_train, y_train_predr),
        mean_squared_error(y_test, y_test_predr)))


print('R^2 train: %.3f, test3: %.3f' % (
        r2_score(y_train, y_train_predr),
        r2_score(y_test, y_test_predr)))

plt.scatter(y_train_predr,  y_train_predr - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_predr,  y_test_predr - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values using ridge')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()

print("My name is Xuebin Wang")
print("My NetID is: xuebinw2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


