# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

os.chdir("C:\TREES")

#Load the dataset

AH_data = pd.read_csv("health.csv")

data_clean = AH_data.dropna()

data_clean.dtypes
print(data_clean.describe())


predictors = data_clean[['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN',
'age','ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail','DEP1',
'ESTEEM1','VIOL1','PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT','PARACTV',
'PARPRES']]

targets = data_clean.TREG1

pred_train, pred_test, tar_train, tar_test=train_test_split(predictors, targets, test_size=.4)
print(pred_train.shape)
print(pred_test.shape)
print(tar_train.shape)
print(tar_test.shape)



classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

print(sklearn.metrics.confusion_matrix(tar_test,predictions))
print(sklearn.metrics.accuracy_score(tar_test, predictions))



#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
with open("fruit_classifier.txt", "w") as f:
    f = tree.export_graphviz(classifier, out_file=f)

