import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import imblearn


def SVM_classify(X : np.array, lbl : np.array):
    #divide into training and test set using sklearn
    from sklearn.model_selection import train_test_split, cross_val_score
    X_train, X_test, lbl_train, lbl_test = train_test_split(X, lbl, test_size = 0.20)  

    #apply random undersampling
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    rus = RandomUnderSampler(return_indices=True)
    X_train_rus, lbl_train_rus, id_rus = rus.fit_sample(X_train, lbl_train)
    # X_train_rus = X_train
    # lbl_train_rus = lbl_train

    #import and train SVC (classifier) 
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='poly', degree=10, gamma='auto')
    svclassifier.fit(X_train_rus, lbl_train_rus)
    #predict labels for test data
    lbl_pred = svclassifier.predict(X_test)
    #evaluate results using built in report and confusion matrix
    from sklearn.metrics import classification_report, confusion_matrix  
    print(confusion_matrix(lbl_test,lbl_pred))  
    print(classification_report(lbl_test,lbl_pred))
    print(lbl_test)
    print(lbl_pred)
    scores = cross_val_score(svclassifier, X, lbl, cv=5)
    print(scores.mean())