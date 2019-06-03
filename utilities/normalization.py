import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

fname = "../data/20171101_RAW_export.xyz"
from data_reader import load_data
df = load_data(fname)
dbdt = df.loc[:,'DBDT_Ch2GT8':'DBDT_Ch2GT20'].values
lbl = df.loc[:,'DBDT_INUSE_Ch2GT14'].values

#normalize
dbdt_scaler = preprocessing.StandardScaler().fit(dbdt)
dbdt_std = dbdt_scaler.transform(dbdt)
X = dbdt_std

#apply random undersampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
rus = RandomOverSampler(return_indices=True)
X_rus, lbl_rus, id_rus = rus.fit_sample(X, lbl)

#divide into training and test set using sklearn
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, lbl_train, lbl_test = train_test_split(X_rus, lbl_rus, test_size = 0.20)  
#import and train SVC (classifier) 
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear', gamma='auto')  
svclassifier.fit(X_train, lbl_train)
#predict labels for test data
lbl_pred = svclassifier.predict(X_test)  
#evaluate results using built in report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(X_test.shape)
print(confusion_matrix(lbl_test,lbl_pred))  
print(classification_report(lbl_test,lbl_pred))
print(lbl_test)
print(lbl_pred)
scores = cross_val_score(svclassifier, X, lbl, cv=5)
print(scores.mean())
