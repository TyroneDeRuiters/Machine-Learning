#Tutorial used: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#Reading in the data from the file.
data = pd.read_csv("breast-cancer-wisconsin-with-headings.txt", header = 0)
data = data.dropna()

#Observing data 
#print(data.shape)
#print(list(data.columns))
#print(data.head()) #if you wish to view the first few lines of the data
#print(data['y'].value_counts()) # returns a value-count for each cancer type
#sns.countplot(x = 'y', data = data, palette = 'hls')
#plt.show()
#plt.savefig('count_plot')
#print(data.groupby('y').mean()) # Here the average of each feature is compared by cancer type

#implementing RFE. This is an elimation process that eliminates unecessary features.
data_vars=data.columns.values.tolist()
y=['y']
X=[i for i in data_vars if i not in y]
#from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 9)
rfe = rfe.fit(data[X], data[y].values.ravel())
print(rfe.support_)
print(rfe.ranking_) # All features ranked 1 are important to the model

#Model summary
#import statsmodels.api as sm
#logit_model=sm.Logit(y,X)
#result=logit_model.fit()
#print(result.summary())

#setting training and validation data set ratios
X = data[X]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,train_size=0.8, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#predicting results and calculating logistic regression accuracy
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#cross-validation
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

